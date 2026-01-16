# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
import time
import re
import glob
import requests
from llm_client import LLMClient

def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    cfg_path = os.path.join(_script_dir(), "json", "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_target_file(filename):
    """查找指定的文件，如果未指定则查找最新的"""
    cleandata_dir = os.path.join(_script_dir(), "cleandata")
    if filename:
        path = os.path.join(cleandata_dir, filename)
        if os.path.exists(path):
            return path
        else:
            print(f"未找到指定文件: {path}")
            return None
    
    # 查找最新的 test_cleaning_result
    files = glob.glob(os.path.join(cleandata_dir, "test_cleaning_result_*.xlsx"))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def infer_scene_name(scene_id, subjects, client, model):
    """
    调用 LLM 推测场景名称
    """
    # 限制 subjects 数量和长度，避免 token 溢出
    sample_subjects = subjects[:20] 
    content_text = "\n".join([f"- {str(s)[:100]}" for s in sample_subjects])
    
    prompt = f"""
你是一个游戏场景分析专家。以下是一组来自游戏场景 ID "{scene_id}" 的玩家反馈/Bug描述。
请根据这些描述的内容，推测该场景在游戏中可能显示的中文名称（例如："主城", "新手村", "火焰山副本", "竞技场" 等）。

请只返回一个最可能的场景名称，不要包含任何解释、标点符号或其他文字。如果无法推测，请返回 "未知场景"。

反馈列表：
{content_text}
"""
    try:
        messages = [{"role": "user", "content": prompt}]
        # 使用配置中的 review_model (通常较快) 或 model
        response = client.chat_completion(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=20
        )
        return response.strip()
    except Exception as e:
        print(f"LLM 推测失败 (Scene {scene_id}): {e}")
        return "分析失败"

def main():
    print("=" * 50)
    print("    场景名称分析工具 (Scene Name Analyzer)    ")
    print("=" * 50)
    
    # 1. 加载配置
    cfg = load_config()
    llm_cfg = cfg.get("llm_settings", {})
    if not llm_cfg.get("enabled", False):
        print("LLM 功能未启用，无法进行分析。")
        return

    # 初始化 LLM Client
    client = LLMClient(config_path=os.path.join(_script_dir(), "json", "llm_config.json"))
    # 如果 llm_config.json 不存在或不完整，可能需要手动设置
    # 这里为了保险，直接使用 config.json 里的配置来初始化/覆盖
    # 但 LLMClient 的设计是读取 llm_config.json，或者我们可以临时构造一个配置对象
    # 让我们直接使用 llm_client.py 的方式，或者手动调用 requests (如果 LLMClient 不好用)
    # 查看 llm_client.py 源码发现它依赖 llm_config.json。
    # 为了简单起见，且保持与 AI_analyze_data.py 一致，我们直接使用 requests 调用，复用配置。
    
    # 修正: 直接复用 analyze_subjects_with_llm 的调用逻辑，或者简单封装
    app_id = llm_cfg.get("app_id", "")
    app_key = llm_cfg.get("app_key", "")
    base_url = llm_cfg.get("base_url", "").rstrip("/")
    # 使用 review_model 进行快速分析，如果没有则用 model
    model = llm_cfg.get("model", llm_cfg.get("model", "glm-4.5-flash"))
    
    print(f"使用模型: {model}")
    
    # 2. 加载数据
    target_filename = "test_cleaning_result_20260115_162609.xlsx"
    file_path = find_target_file(target_filename)
    
    # 如果指定文件不存在，尝试找最新的
    if not file_path:
        print(f"文件 {target_filename} 不存在，尝试查找最新的测试结果...")
        file_path = find_target_file(None)
        
    if not file_path:
        print("未找到任何测试结果文件。请先运行 main_test.py 的模式 4。")
        return
        
    print(f"正在读取文件: {file_path}")
    df = pd.read_excel(file_path)
    
    if 'scene' not in df.columns or 'subject' not in df.columns:
        print("数据缺少 'scene' 或 'subject' 列。")
        return
        
    # 3. 分组分析
    # 过滤掉 scene 为空的
    df_valid = df[df['scene'].notna() & (df['scene'].astype(str).str.strip() != "")].copy()
    unique_scenes = df_valid['scene'].unique()
    
    print(f"共发现 {len(unique_scenes)} 个有效场景 ID。开始分析...")
    
    scene_names = {}
    
    # 简单的 requests 封装
    import requests
    from difflib import SequenceMatcher
    
    auth_key = f"{app_id}.{app_key}" if app_id else app_key
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }
    
    # 尝试引入 tqdm
    try:
        from tqdm import tqdm
        iter_scenes = tqdm(unique_scenes, desc="场景名称推测中")
    except ImportError:
        iter_scenes = unique_scenes
        print("建议安装 tqdm 以显示进度条: pip install tqdm")

    # 用于去重的缓存: name -> (count, scene_id)
    # 这里我们只做简单的去重合并：如果新推测的名字和已有的重复度 > 30%，
    # 则比较两者对应的条目数，保留条目数多的那个名字作为统一名称。
    # 但注意：这里是 map scene_id -> name，我们不能修改 key (scene_id)，只能修改 value (name)
    # 所以逻辑是：推测出 name 后，去 checked_names 里找相似的。
    # 如果找到相似度 > 0.3 的 existing_name：
    #    比较 current_scene_count 和 existing_scene_count
    #    如果 current > existing: 
    #       current_name 胜出，update existing_scene_id's name to current_name
    #       record current_name -> current_count
    #    else:
    #       existing_name 胜出，use existing_name for current_scene_id
    #       update record count
    
    checked_names = {} # name -> {"count": count, "scene_ids": [id1, id2...]}
    
    for scene_id in iter_scenes:
        scene_subjects = df_valid[df_valid['scene'] == scene_id]['subject'].tolist()
        # 简单去重
        scene_subjects = list(set(scene_subjects))
        count = len(scene_subjects)
        
        name = infer_scene_name_requests(scene_id, scene_subjects, base_url, headers, model)
        
        # --- 简易去重机制 ---
        best_match_name = None
        best_ratio = 0.0
        
        for existing_name in checked_names.keys():
            ratio = SequenceMatcher(None, name, existing_name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_name = existing_name
        
        final_name = name
        
        if best_ratio > 0.3: # 阈值 30%
            # 发现相似，进行合并判定
            existing_info = checked_names[best_match_name]
            existing_count = existing_info["count"]
            
            if count > existing_count:
                # 当前场景条目更多，以当前名字为准
                # 更新已有的场景ID
                for sid in existing_info["scene_ids"]:
                    scene_names[sid] = name
                
                # 更新 checked_names
                # 删除旧的 key，添加新的 key (或者合并)
                del checked_names[best_match_name]
                existing_info["count"] += count
                existing_info["scene_ids"].append(scene_id)
                checked_names[name] = existing_info
                final_name = name
            else:
                # 已有场景条目更多，以已有名字为准
                final_name = best_match_name
                existing_info["count"] += count
                existing_info["scene_ids"].append(scene_id)
        else:
            # 无相似，新增
            checked_names[name] = {"count": count, "scene_ids": [scene_id]}
            
        scene_names[scene_id] = final_name
        
        # 避免速率限制
        time.sleep(0.2)
        
    # 4. 回填结果
    print("\n正在回填结果...")
    df['scene name'] = df['scene'].map(scene_names)
    
    # 5. 保存
    output_path = file_path.replace(".xlsx", "_analyzed.xlsx")
    df.to_excel(output_path, index=False)
    print(f"结果已保存至: {output_path}")

def infer_scene_name_requests(scene_id, subjects, base_url, headers, model):
    # 限制 subjects 数量和长度
    sample_subjects = subjects[:15] 
    content_text = "\n".join([f"- {str(s)[:100]}" for s in sample_subjects])
    
    prompt = f"""
你是一个游戏场景分析专家。以下是一组来自游戏场景 ID "{scene_id}" 的玩家反馈/Bug描述。
请根据这些描述的内容，推测该场景在游戏中可能显示的中文名称（例如："主城", "新手村", "火焰山副本", "竞技场" 等）。

请只返回一个最可能的场景名称，不要包含任何解释、标点符号或其他文字。如果无法推测，请返回 "未知场景"。

反馈列表：
{content_text}
"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        content = res_json['choices'][0]['message']['content'].strip()
        # 清理可能的多余字符
        content = content.replace("场景名称：", "").replace("场景名：", "").strip()
        return content
    except Exception as e:
        return f"Error: {str(e)[:20]}"

if __name__ == "__main__":
    main()
