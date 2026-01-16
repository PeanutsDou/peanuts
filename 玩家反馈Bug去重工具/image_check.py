# -*- coding: utf-8 -*-
import os
import pandas as pd
import glob
import json
import random
import time
import argparse
import re
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def _script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    """加载配置文件"""
    cfg_path = os.path.join(_script_dir(), "json", "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_image_column(df):
    """
    查找可能的图片列名
    """
    cols = list(df.columns)
    for col in cols:
        name = str(col).lower()
        if ("image" in name) or ("img" in name) or ("图片" in str(col)) or ("截图" in str(col)) or ("pic" in name) or ("photo" in name) or ("链接" in str(col)) or ("url" in name):
            return col
    return None

def fallback_judge(text):
    """
    兜底判断逻辑：当 LLM 调用失败时使用规则判断
    """
    s = str(text or "").strip()
    if not s:
        return True, "描述为空或仅包含空白，需看图"
    
    # 去除特殊符号计算长度
    content = re.sub(r"[\s\W_]+", "", s)
    length = len(content)
    
    # 关键词匹配
    vague_keywords = r"(卡|卡顿|慢|不行|不对|有问题|坏了|闪退|崩|卡死|bug|异常|怎么回事|这啥)"
    location_keywords = r"(地图|副本|npc|道具|装备|技能|任务|按钮|界面|设置|分辨率|帧率|服务器|时间|坐标|位置|名字|点击|显示)"
    
    vague = re.search(vague_keywords, s.lower())
    has_location = re.search(location_keywords, s.lower())
    
    if length <= 10 and (not has_location):
        return True, "字数极少且信息模糊/缺定位信息，需看图"
    
    return False, "描述较清晰或信息量充足，不需要看图"

def call_llm_image_check(items, config):
    """
    调用 LLM 批量判断是否需要看图
    items: [{"id": ..., "subject": ...}, ...]
    返回: ({id: {"needs_image": bool, "reason": str}}, input_chars, output_chars)
    """
    llm_cfg = config.get("llm_settings", {})
    app_id = llm_cfg.get("app_id", "")
    app_key = llm_cfg.get("app_key", "")
    base_url = llm_cfg.get("base_url", "").rstrip("/")
    # 使用基础模型，通常足够了
    model = llm_cfg.get("model", "glm-4.5-flash") 
    timeout = llm_cfg.get("timeout", 120)
    
    input_chars = 0
    output_chars = 0
    
    if not app_key:
        print("错误: LLM app_key 未配置。")
        return {}, 0, 0
        
    auth_key = f"{app_id}.{app_key}" if app_id else app_key
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }
    
    prompt = """
判断Bug描述是否需查看截图辅助定位。

标准(严格克制)：
1. 必须看图：描述为空、"如图"、"看图"或缺乏关键定位信息(无地图/NPC/道具/时间)。
2. 不需要看图：描述清晰，含具体位置、名称或详细现象。

示例：
- "如图" -> {"needs_image": true, "reason": "完全依赖图片"}
- "玩着玩着卡死了" -> {"needs_image": true, "reason": "缺乏场景定位"}
- "主城铁匠对话后卡死" -> {"needs_image": false, "reason": "定位清晰"}

Input List:
"""
    for item in items:
        # 限制长度防止超 token
        subj = str(item['subject'])[:200].replace("\n", " ")
        prompt += f"ID_{item['id']}: {subj}\n"
        
    prompt += """
Output (JSON only):
{
  "ID_x": {"needs_image": true/false, "reason": "简短理由"},
  ...
}
"""
    input_chars = len(prompt)
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"} # 如果支持的话
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            output_chars = len(content)
            
            # 清理 Markdown
            clean_content = re.sub(r'^```json\s*', '', content.strip())
            clean_content = re.sub(r'^```\s*', '', clean_content)
            clean_content = re.sub(r'\s*```$', '', clean_content)
            
            parsed = json.loads(clean_content)
            
            # 格式化 ID 键值
            result = {}
            for k, v in parsed.items():
                # 尝试提取 ID
                m = re.search(r"ID_(\w+)", str(k))
                if m:
                    raw_id = m.group(1)
                    # 尝试转回原始类型（通常是 int 或 str）
                    # 这里我们在 items 里存的是原始 id，所以尽量匹配
                    # 为了通用，我们在 items 里把 id 转为了 str 使用
                    result[str(raw_id)] = v
                else:
                    # 尝试直接匹配
                    result[str(k)] = v
            return result, input_chars, output_chars

        except Exception as e:
            # print(f"LLM Image Check 失败 (尝试 {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                
    return {}, input_chars, 0

def run_image_check_batch(items_list, config):
    """
    对外接口：批量检测
    items_list: [{"id": ..., "subject": ...}, ...]
    返回: (all_results, total_in_chars, total_out_chars)
    """
    # 分批处理
    batch_size = 20
    batches = [items_list[i:i + batch_size] for i in range(0, len(items_list), batch_size)]
    
    all_results = {}
    total_in = 0
    total_out = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(call_llm_image_check, batch, config): batch for batch in batches}
        
        # 尝试引入 tqdm 显示进度
        try:
            from tqdm import tqdm
            iter_futures = tqdm(as_completed(futures), total=len(futures), desc="AI图片需求分析")
        except ImportError:
            iter_futures = as_completed(futures)
            print("提示: 安装 tqdm 可显示进度条 (pip install tqdm)")

        for future in iter_futures:
            try:
                # call_llm_image_check 现在返回 (result_dict, in_chars, out_chars)
                batch_res, in_c, out_c = future.result()
                all_results.update(batch_res)
                total_in += in_c
                total_out += out_c
            except Exception as e:
                print(f"Image Check Batch Error: {e}")
                
    return all_results, total_in, total_out

