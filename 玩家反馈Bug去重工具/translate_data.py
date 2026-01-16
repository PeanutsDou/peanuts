# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def _script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

# --- Checkpoint 管理 ---
def get_cache_path():
    """获取缓存目录"""
    cache_dir = os.path.join(_script_dir(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

CHECKPOINT_FILE = "normalization_checkpoint.json"

def load_checkpoint():
    """加载 Checkpoint (Subject -> Data)"""
    cp_path = os.path.join(get_cache_path(), CHECKPOINT_FILE)
    if os.path.exists(cp_path):
        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取 Checkpoint 失败: {e}")
    return {}

def save_checkpoint(data):
    """保存 Checkpoint"""
    cp_path = os.path.join(get_cache_path(), CHECKPOINT_FILE)
    try:
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存 Checkpoint 失败: {e}")

def load_config():
    """加载配置文件"""
    cfg_path = os.path.join(_script_dir(), "json", "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def call_llm_normalize(subjects, config):
    """
    调用 LLM 批量对 Bug 描述进行规范化
    :param subjects: 原始描述列表
    :param config: 配置字典
    :return: (results, prompt_tokens, completion_tokens)
    """
    llm_cfg = config.get("llm_settings", {})
    app_id = llm_cfg.get("app_id", "")
    app_key = llm_cfg.get("app_key", "")
    base_url = llm_cfg.get("base_url", "").rstrip("/")
    model = llm_cfg.get("model", "glm-4.5-flash")
    timeout = llm_cfg.get("timeout", 120)
    
    if not app_key:
        print("错误: LLM app_key 未配置。")
        # 返回空结果和0消耗
        return [{"original": s, "normalized": [s]} for s in subjects], 0, 0
        
    auth_key = f"{app_id}.{app_key}" if app_id else app_key
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }
    
    # 构造 Prompt
    content_text = "\n".join([f"{i+1}. {str(s)[:500]}" for i, s in enumerate(subjects)])
    
    prompt = f"""
你是一名资深游戏测试工程师。请对玩家反馈的Bug描述进行**规范化重写**和**结构化提取**。

**核心原则：**
1. **客观陈述**：去除"恶心"、"无语"等情绪化词汇，仅保留事实。
2. **要素明确**：描述须包含【主体】(什么东西)、【位置】(在哪)、【现象】(怎么了)。
3. **独立拆分**：若一条反馈包含多个独立问题，必须拆分为多条。
4. **精简补全**：补全缺失主语(如"卡了"->"游戏画面卡顿")，删除冗余助词，严禁臆测。

**示例参考：**

*案例1 (情绪化/冗余 -> 规范化)*
输入："这个冲锋号我有了，领不了，而且红点消不掉，影响游戏体验，总想把红点点掉"
输出：
[
  {{
    "text": "已拥有冲锋号但无法领取，且红点提示无法消除",
    "core": "冲锋号/红点",
    "loc": "奖励领取界面",
    "phen": "无法领取/红点无法消除"
  }}
]

*案例2 (简单描述 -> 规范化)*
输入："意识防护这个芯片的特效过于亮眼希望可以降低一些"
输出：
[
  {{
    "text": "意识防护芯片特效亮度过高",
    "core": "意识防护芯片",
    "loc": "技能特效",
    "phen": "亮度过高"
  }}
]

*案例3 (多问题拆分)*
输入："水下工厂传送带不刷怪，万国酒店九楼泳池没有水，傀儡迷途打一半时会特别卡，任何指令都要三秒后生效"
输出：
[
  {{
    "text": "水下工厂传送带不刷新怪物",
    "core": "传送带",
    "loc": "水下工厂",
    "phen": "不刷怪"
  }},
  {{
    "text": "万国酒店九楼泳池缺失水体",
    "core": "泳池",
    "loc": "万国酒店九楼",
    "phen": "缺失水体"
  }},
  {{
    "text": "傀儡迷途战斗中严重卡顿且指令延迟",
    "core": "游戏进程",
    "loc": "傀儡迷途",
    "phen": "卡顿/指令延迟"
  }}
]

**任务要求：**
请直接返回JSON列表，列表元素包含：
- "id": 输入序号
- "normalized": 对象列表，每个对象含 "text"(规范描述), "core"(主体), "loc"(位置), "phen"(现象)

**输入列表：**
{content_text}
"""
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"}
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            
            # 获取 Token 消耗
            usage = res_json.get("usage", {})
            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)
            
            # 清理 Markdown
            clean_content = re.sub(r'^```json\s*', '', content.strip())
            clean_content = re.sub(r'^```\s*', '', clean_content)
            clean_content = re.sub(r'\s*```$', '', clean_content)
            
            parsed = json.loads(clean_content)
            
            # 构建结果映射
            result_map = {}
            if isinstance(parsed, list):
                # 某些模型可能直接返回 list
                for item in parsed:
                    idx = item.get("id")
                    norm = item.get("normalized")
                    if idx is not None and norm:
                        result_map[idx] = norm
            elif isinstance(parsed, dict):
                # 某些模型可能包在一层 key 里，或者直接就是 list
                for k, v in parsed.items():
                    if isinstance(v, list):
                        for item in v:
                            idx = item.get("id")
                            norm = item.get("normalized")
                            if idx is not None and norm:
                                result_map[idx] = norm
                        break
            
            # 按顺序重组
            final_results = []
            for i, subj in enumerate(subjects):
                idx = i + 1
                if idx in result_map:
                    final_results.append({"original": subj, "normalized": result_map[idx]})
                else:
                    # 回退：没有结构化信息
                    final_results.append({
                        "original": subj, 
                        "normalized": [{"text": subj, "core": "", "loc": "", "phen": ""}]
                    })
            
            return final_results, p_tokens, c_tokens

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # 失败回退
    return [{
        "original": s, 
        "normalized": [{"text": s, "core": "", "loc": "", "phen": ""}]
    } for s in subjects], 0, 0

def run_normalization(df, skip_norm=False, skip_img=False):
    """
    执行规范化流程 (Checkpoint 版)
    :param df: 输入的 DataFrame
    :param skip_norm: 是否跳过语义规范化
    :param skip_img: 是否跳过图片检查
    :return: (new_df, translation_logs_df, image_logs_df)
    """
    if df is None or df.empty:
        return df, None, None
        
    start_time = time.time()
    cfg = load_config()
    
    # 检查是否启用 LLM
    llm_settings = cfg.get("llm_settings", {})
    if not llm_settings.get("enabled", False):
        print("LLM 功能未启用，跳过规范化。")
        return df, None, None

    if 'subject' not in df.columns:
        print("未找到 subject 列，跳过规范化。")
        return df, None, None
    
    # 加载 Checkpoint
    checkpoint = load_checkpoint()
    print(f"已加载 Checkpoint，包含 {len(checkpoint)} 条历史数据。")

    # 统计 Token 消耗
    total_input_chars = 0
    total_output_chars = 0
    
    # 1. 识别需要规范化的 Unique Subject
    if not skip_norm:
        all_subjects = df['subject'].astype(str).unique().tolist()
        missing_normalization = [s for s in all_subjects if s not in checkpoint or 'normalized' not in checkpoint[s]]
        
        if missing_normalization:
            print(f"发现 {len(missing_normalization)} 条新数据需要规范化...")
            batch_size = 20
            batches = [missing_normalization[i:i + batch_size] for i in range(0, len(missing_normalization), batch_size)]
            
            # 并发处理
            max_workers = 5
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(batches), desc="LLM 规范化进度")
            except ImportError:
                pbar = None
                
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(call_llm_normalize, batch, cfg): batch for batch in batches}
                
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        res, in_c, out_c = future.result()
                        total_input_chars += in_c
                        total_output_chars += out_c
                        
                        # 更新 Checkpoint (内存)
                        for item in res:
                            orig = item['original']
                            if orig not in checkpoint:
                                checkpoint[orig] = {}
                            checkpoint[orig]['normalized'] = item['normalized']
                            
                    except Exception as e:
                        print(f"Batch 处理失败: {e}")
                    
                    if pbar: pbar.update(1)
            
            if pbar: pbar.close()
            
            # 保存一次 Checkpoint
            save_checkpoint(checkpoint)
        else:
            print("所有数据均已在 Checkpoint 中找到，跳过规范化 API 调用。")
    else:
        print(">>> 已跳过 [语义规范化] 步骤。")

    # 2. 图片检测 (Image Check)
    if not skip_img:
        print(">>> 子步骤: 图片必要性检测 (Image Check) <<<")
        from image_check import find_image_column, run_image_check_batch, fallback_judge
        img_col = find_image_column(df)
        
        # 结果映射: subject -> (needs_image, reason)
        # 注意: 这里只存储 LLM 的判断结果，不考虑具体行是否有图片链接
        
        if img_col:
            print(f"检测到图片列: {img_col}")
            
            # 找出那些【有图片链接】且【Checkpoint中没有图片检测结果】的 Subject
            subjects_needing_img_check = []
            seen_subj = set()
            
            for idx, row in df.iterrows():
                subj = str(row['subject'])
                if subj in seen_subj: continue
                
                # 检查该行是否有图片
                img_val = row.get(img_col)
                has_img = pd.notna(img_val) and str(img_val).strip()
                
                if has_img:
                    # 检查 Checkpoint 是否已有结果
                    if subj not in checkpoint or 'image_check' not in checkpoint[subj]:
                        subjects_needing_img_check.append(subj)
                        seen_subj.add(subj)
            
            if subjects_needing_img_check:
                print(f"发现 {len(subjects_needing_img_check)} 条含图数据需要 LLM 判断...")
                # 构造 run_image_check_batch 需要的格式 [{"id": hash, "subject": subj}, ...]
                # 使用 MD5 哈希作为 ID，防止特殊字符导致 ID 解析失败
                import hashlib
                id_map = {} # hash -> subj
                items_to_check = []
                for s in subjects_needing_img_check:
                    s_id = hashlib.md5(s.encode('utf-8')).hexdigest()
                    id_map[s_id] = s
                    items_to_check.append({"id": s_id, "subject": s})
                
                # 批量调用
                check_results_raw = run_image_check_batch(items_to_check, cfg)
                
                check_results = {}
                if isinstance(check_results_raw, tuple):
                     check_results = check_results_raw[0]
                     total_input_chars += check_results_raw[1]
                     total_output_chars += check_results_raw[2]
                else:
                     check_results = check_results_raw
                
                # 更新 Checkpoint
                for s_id, res in check_results.items():
                    # 映射回原始 subject
                    subj = id_map.get(s_id)
                    if not subj:
                        # 兼容性处理：如果 ID 没在 map 里，可能是 LLM 返回了原始 subject (极少情况)
                        # 或者之前的逻辑遗留
                        continue

                    if subj not in checkpoint:
                        checkpoint[subj] = {}
                    checkpoint[subj]['image_check'] = {
                        "needs_image": res.get("needs_image", False),
                        "reason": res.get("reason", "")
                    }
                
                save_checkpoint(checkpoint)
            else:
                print("无需新增图片检测 (无新含图数据 或 已缓存)。")
                
        else:
            print("未检测到图片列，跳过图片检测。")
    else:
        print(">>> 已跳过 [图片查看分析] 步骤。")
        # 需要定义 img_col 和 fallback_judge，防止后面引用报错
        from image_check import find_image_column, fallback_judge
        img_col = find_image_column(df)

    # 3. 重组 DataFrame (使用 Checkpoint 数据)
    new_rows = []
    log_rows = []
    img_log_rows = []
    
    for idx, row in df.iterrows():
        orig_subj = str(row['subject'])
        
        # 获取规范化结果
        norm_data = []
        if not skip_norm:
            norm_data = checkpoint.get(orig_subj, {}).get('normalized', [])
        
        if not norm_data:
            # 兜底：未找到缓存 或 跳过规范化
            norm_data = [{"text": orig_subj, "core": "", "loc": "", "phen": ""}]
            
        # 记录日志
        norm_str_list = []
        for n in norm_data:
            if isinstance(n, str): norm_str_list.append(n)
            else: norm_str_list.append(n.get("text", ""))
            
        log_rows.append({
            "original_subject": orig_subj,
            "normalized_subject": " | ".join(norm_str_list) 
        })
        
        # 获取图片检测结果 (LLM 意见)
        # 只有当行里真的有图片时，才应用 LLM 的意见；否则是 False (无图片)
        img_val = row.get(img_col) if img_col else None
        has_img_link = pd.notna(img_val) and str(img_val).strip()
        
        if has_img_link:
            if not skip_img:
                img_check_data = checkpoint.get(orig_subj, {}).get('image_check', {})
                # 如果没有缓存结果 (可能是 API 失败)，兜底判断
                if not img_check_data:
                     ni, reas = fallback_judge(orig_subj)
                     img_check_data = {"needs_image": ni, "reason": reas}
                
                needs_img = img_check_data.get("needs_image", False)
                reason = img_check_data.get("reason", "")
            else:
                # 跳过图片检测，默认为不看图
                needs_img = False
                reason = "用户跳过图片检测"
        else:
            needs_img = False
            reason = "无图片链接" if img_col else "无图片列"

        # 记录图片日志
        if img_col:
            task_id = row.get('task_id', idx)
            img_log_rows.append({
                "task_id": task_id,
                "subject": orig_subj,
                "needs_image": "Yes" if needs_img else "No",
                "reason": reason
            })

        # 创建新行 (拆分)
        for item_norm in norm_data:
            if isinstance(item_norm, str):
                norm_text = item_norm
                core, loc, phen = "", "", ""
            else:
                norm_text = item_norm.get("text", "")
                core = item_norm.get("core", "")
                loc = item_norm.get("loc", "")
                phen = item_norm.get("phen", "")
            
            if not norm_text: continue
            
            new_row = row.copy()
            if 'original_subject' not in new_row:
                new_row['original_subject'] = orig_subj
            
            new_row['subject'] = str(norm_text).strip()
            new_row['subject_core'] = core
            new_row['location'] = loc
            new_row['phenomenon'] = phen
            
            new_row['needs_image'] = "Yes" if needs_img else "No"
            new_row['needs_image_reason'] = reason
            
            if 'normalized_subject' in new_row:
                del new_row['normalized_subject']
                
            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    translation_logs_df = pd.DataFrame(log_rows)
    image_logs_df = pd.DataFrame(img_log_rows) if img_log_rows else None
    
    end_time = time.time()
    print(f"流程完成。耗时: {end_time - start_time:.2f}秒")
    print(f"Token 消耗参考: 输入 {total_input_chars}, 输出 {total_output_chars}")
    
    return new_df, translation_logs_df, image_logs_df
