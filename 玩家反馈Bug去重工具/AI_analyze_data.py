# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import requests
import time
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    cfg_path = os.path.join(_script_dir(), "json", "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_latest_dedup_file():
    cleandata_dir = os.path.join(_script_dir(), "cleandata")
    if not os.path.isdir(cleandata_dir):
        return None
    files = [
        os.path.join(cleandata_dir, f)
        for f in os.listdir(cleandata_dir)
        if f.endswith('.xlsx') and not f.startswith('~$') and '_AI_Analyzed' not in f
    ]
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def analyze_subjects_with_llm(subjects, config):
    """
    调用 LLM 对 subjects 进行精细化归类和去重。
    返回结构化数据：
    [
        {
            "representative": "代表条目",
            "items": ["条目1", "条目2"]
        },
        ...
    ]
    """
    llm_cfg = config.get("llm_settings", {})
    app_id = llm_cfg.get("app_id", "")
    app_key = llm_cfg.get("app_key", "")
    base_url = llm_cfg.get("base_url", "").rstrip("/")
    model = llm_cfg.get("model", "glm-4.5-flash")
    timeout = llm_cfg.get("timeout", 120)
    max_retries = llm_cfg.get("max_retries", 3)
    
    if not app_key:
        print("Error: LLM app_key not configured.")
        # Fallback: Treat all as one group? Or all separate?
        # Safe fallback: All separate (no deduplication)
        return [{"representative": s, "items": [s]} for s in subjects]
        
    auth_key = f"{app_id}.{app_key}" if app_id else app_key
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }
    
    # 构造 Prompt
    content_text = "\n".join([f"{i+1}. {str(s)[:500]}" for i, s in enumerate(subjects)]) # Increased limit slightly
    
    prompt = f"""
你是一位资深游戏质量保证(QA)专家。以下是一组玩家反馈的Bug或建议（它们在预处理中被归为一个粗略的聚类）。
请对这些反馈进行精细的语义分析，将它们进一步细分为具体的语义组。
对于每个语义组：
1. 找出该组内描述最完整、信息量最大的一条作为“代表条目”（Representative）。
2. 该组内其他条目视为重复/冗余，将被剔除。

请返回一个 JSON 列表，列表中的每个元素代表一个语义组，包含两个字段：
- "representative": 选出的代表条目原文。
- "items": 该组内包含的所有条目原文列表（包含代表条目本身）。

注意：
- 必须包含输入列表中的所有条目，不要遗漏。
- 如果某条目是独立的（不与其他重复），它自己构成一组，representative 就是它自己，items 也只包含它。
- 保持原文，不要修改文本。
- 请直接输出合法的 JSON 字符串，不要包含 Markdown 格式标记（如 ```json）。

反馈列表：
{content_text}
"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    url = f"{base_url}/chat/completions"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            
            # 清理 Markdown
            clean_content = re.sub(r'^```json\s*', '', content.strip())
            clean_content = re.sub(r'^```\s*', '', clean_content)
            clean_content = re.sub(r'\s*```$', '', clean_content)
            
            parsed_result = json.loads(clean_content)
            
            if isinstance(parsed_result, list):
                # Simple validation
                valid = True
                for item in parsed_result:
                    if not isinstance(item, dict) or 'representative' not in item or 'items' not in item:
                        valid = False
                        break
                if valid:
                    return parsed_result
            
            print(f"LLM returned invalid format (Attempt {attempt+1}): {content[:100]}...")
            
        except Exception as e:
            print(f"LLM call failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                
    # If all retries fail, return original as separate items
    return [{"representative": s, "items": [s]} for s in subjects]

def analyze_single_cluster(args):
    """
    Worker function for processing a single cluster.
    args: (cid, group_df, config, threshold, cohesion_threshold)
    """
    cid, group, config, threshold, cohesion_threshold = args
    
    count = len(group)
    subjects = group['subject'].astype(str).tolist()
    
    # 获取该簇的 Cohesion
    cohesion = 0.0
    if 'cohesion' in group.columns and not group['cohesion'].isna().all():
        cohesion = group['cohesion'].iloc[0]
        
    should_analyze = False
    reasons = []
    
    if count > threshold:
        reasons.append(f"条目数 {count} > {threshold}")
    
    if count > 1 and cohesion < cohesion_threshold:
        reasons.append(f"内聚度 {cohesion:.3f} < {cohesion_threshold}")
        
    cluster_results = []
    cluster_logs = []
    
    if reasons:
        # should_analyze = True
        print(f"  [Cluster {cid}] 需要分析 ({' & '.join(reasons)})，调用 AI...")
        
        # Call LLM
        ai_groups = analyze_subjects_with_llm(subjects, config)
        
        # Process results
        # ai_groups is list of {representative, items}
        
        # Track which subjects were handled
        handled_subjects = set()
        
        for grp in ai_groups:
            rep_subj = grp.get('representative', '')
            items = grp.get('items', [])
            
            if not rep_subj:
                continue
                
            # Find image for representative
            # We look in the original group df for the representative subject
            match_row = group[group['subject'].astype(str) == rep_subj]
            img = ""
            if not match_row.empty:
                img = match_row.iloc[0].get('image', '')
            else:
                # If exact match not found (maybe AI tweaked it slightly or encoding), try to find best match or just pick first from items
                # Or just pick an image from the cluster if we assume they are similar
                # Let's try to find image from the items that are in this group
                found_img = False
                for item_subj in items:
                    m_row = group[group['subject'].astype(str) == item_subj]
                    if not m_row.empty:
                        img_cand = m_row.iloc[0].get('image', '')
                        if img_cand:
                            img = img_cand
                            found_img = True
                            break
                if not found_img:
                     valid_imgs = [x for x in group['image'].tolist() if pd.notna(x) and str(x).strip()]
                     img = valid_imgs[0] if valid_imgs else ""
            
            cluster_results.append({
                "subject": rep_subj,
                "image": img,
                "cluster_id": cid # Keep track of original cluster
            })
            
            # Logs
            for item in items:
                handled_subjects.add(item)
                if item == rep_subj:
                    cluster_logs.append({
                        "cluster_id": cid,
                        "original_subject": item,
                        "action": "kept",
                        "reason": "ai_representative",
                        "final_subject": rep_subj
                    })
                else:
                    cluster_logs.append({
                        "cluster_id": cid,
                        "original_subject": item,
                        "action": "removed",
                        "reason": "ai_merged_into_rep",
                        "final_subject": rep_subj
                    })
                    
        # Check for subjects that were in original but not in AI output (AI Hallucination or omission)
        # If AI omitted them, we should probably keep them or mark as removed?
        # Safe strategy: Keep them as standalone to avoid data loss.
        for s in subjects:
            if s not in handled_subjects:
                 # Try to find if it was renamed? No, assuming exact match.
                 # Add as kept
                 cluster_results.append({
                     "subject": s,
                     "image": group[group['subject'].astype(str) == s].iloc[0].get('image', ''),
                     "cluster_id": cid
                 })
                 cluster_logs.append({
                     "cluster_id": cid,
                     "original_subject": s,
                     "action": "kept",
                     "reason": "ai_omitted_fallback",
                     "final_subject": s
                 })

    else:
        # No analysis needed - Small/Tight cluster
        # Pick one representative (longest)
        representative_row = group.loc[group['subject'].str.len().idxmax()]
        rep_subj = representative_row['subject']
        
        cluster_results.append({
            "subject": rep_subj,
            "image": representative_row.get('image', ''),
            "cluster_id": cid
        })
        
        for _, row in group.iterrows():
            orig_subj = row['subject']
            if orig_subj == rep_subj:
                cluster_logs.append({
                    "cluster_id": cid,
                    "original_subject": orig_subj,
                    "action": "kept",
                    "reason": "small_cluster_rep",
                    "final_subject": rep_subj
                })
            else:
                cluster_logs.append({
                    "cluster_id": cid,
                    "original_subject": orig_subj,
                    "action": "removed",
                    "reason": "small_cluster_folded",
                    "final_subject": rep_subj
                })
                
    return cid, cluster_results, cluster_logs

def run_ai_analysis():
    print("\n=== 步骤 3: AI 语义分析与最终去重 ===")
    
    cfg = load_config()
    llm_settings = cfg.get("llm_settings", {})
    if not llm_settings.get("enabled", False):
        print("LLM 分析功能未启用，跳过。")
        return
        
    threshold = llm_settings.get("ai_analyze_threshold", 5)
    cohesion_threshold = llm_settings.get("ai_cohesion_threshold", 0.9)
    
    target_file = find_latest_dedup_file()
    if not target_file:
        print("未找到可处理的去重结果文件。")
        return
        
    print(f"正在处理文件: {target_file}")
    
    try:
        xls = pd.ExcelFile(target_file)
        df = pd.read_excel(xls, sheet_name='Final_Deduplicated')
        
        existing_sheets = {}
        for sheet_name in xls.sheet_names:
            if sheet_name != 'Final_Deduplicated':
                existing_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
                
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    if 'cluster_id' not in df.columns:
        print("数据中缺少 cluster_id 列，无法进行簇内分析。")
        return

    final_results = []
    ai_logs = []
    
    # 1. 处理噪点 (cluster_id == -1)
    df_noise = df[df['cluster_id'] == -1]
    print(f"保留噪点数据: {len(df_noise)} 条")
    for _, row in df_noise.iterrows():
        final_results.append({
            "subject": row['subject'],
            "image": row.get('image', '')
        })
        ai_logs.append({
            "cluster_id": -1,
            "original_subject": row['subject'],
            "action": "kept",
            "reason": "noise",
            "final_subject": row['subject']
        })
        
    # 2. 并发处理聚类
    grouped = df[df['cluster_id'] != -1].groupby('cluster_id')
    groups_to_process = []
    for cid, group in grouped:
        groups_to_process.append((cid, group, cfg, threshold, cohesion_threshold))
        
    print(f"开始分析 {len(groups_to_process)} 个聚类 (并发模式)...")
    
    # 并发执行
    # Max workers can be tuned. Since it's network bound, we can go higher, but let's be conservative to avoid hitting API limits too hard.
    max_workers = 5 
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_cluster, args): args[0] for args in groups_to_process}
        
        completed_count = 0
        total_count = len(groups_to_process)
        
        for future in as_completed(futures):
            cid = futures[future]
            try:
                _, c_results, c_logs = future.result()
                final_results.extend(c_results)
                ai_logs.extend(c_logs)
            except Exception as e:
                print(f"Cluster {cid} 处理异常: {e}")
            
            completed_count += 1
            if completed_count % 5 == 0:
                print(f"    进度: {completed_count}/{total_count}")

    # 3. 生成最终表格
    # 基于 AI 日志与前序语义去重日志，构建统一的 Remove_Logs 记录
    remove_logs_records = []

    # 3.1 旧日志
    existing_remove_df = existing_sheets.get('Removed_Logs')
    if existing_remove_df is not None:
        for _, row in existing_remove_df.iterrows():
            removed_subj = row.get('removed') if 'removed' in existing_remove_df.columns else row.get('subject')
            if pd.isna(removed_subj):
                continue
            stage_val = row.get('stage', 'semantic_dedup') if 'stage' in existing_remove_df.columns else 'semantic_dedup'
            reason_val = row.get('similarity', '') if 'similarity' in existing_remove_df.columns else ''
            kept_val = row.get('kept', '') if 'kept' in existing_remove_df.columns else ''
            remove_logs_records.append({
                "subject": str(removed_subj),
                "stage": str(stage_val),
                "reason": str(reason_val) if not pd.isna(reason_val) else "",
                "representative_subject": str(kept_val) if not pd.isna(kept_val) else ""
            })

    # 3.2 新日志
    for log in ai_logs:
        if log.get("action") == "removed":
            remove_logs_records.append({
                "subject": str(log.get("original_subject")),
                "stage": "ai_analyze",
                "reason": str(log.get("reason")),
                "representative_subject": str(log.get("final_subject"))
            })

    df_remove_logs = None
    if remove_logs_records:
        df_remove_logs = pd.DataFrame(remove_logs_records)
        cols_remove = ["subject", "stage", "reason", "representative_subject"]
        df_remove_logs = df_remove_logs[cols_remove]

    df_final_output = pd.DataFrame(final_results)
    cols = ['subject', 'image']
    if 'image' not in df_final_output.columns:
        df_final_output['image'] = ""
    df_final_output = df_final_output[cols]
    
    output_filename = os.path.basename(target_file).replace(".xlsx", "_AI_Analyzed.xlsx")
    output_path = os.path.join(os.path.dirname(target_file), output_filename)
    
    is_test = cfg.get("isTest", False)
    
    if is_test:
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_final_output.to_excel(writer, sheet_name='Final_Result', index=False)
                
                if df_remove_logs is not None and not df_remove_logs.empty:
                    df_remove_logs.to_excel(writer, sheet_name='Removed_Logs', index=False)

                if ai_logs:
                    df_ai_logs = pd.DataFrame(ai_logs)
                    df_ai_logs.to_excel(writer, sheet_name='AI_Analysis_Logs', index=False)
                
                for sheet_name, df_sheet in existing_sheets.items():
                    if sheet_name == 'Removed_Logs':
                        continue
                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                    
        except PermissionError:
            print(f"错误: 无法写入文件 {output_path}。文件可能被打开。")
            new_output_path = output_path.replace(".xlsx", f"_{int(time.time())}.xlsx")
            print(f"尝试保存到新文件: {new_output_path}")
            try:
                with pd.ExcelWriter(new_output_path, engine='openpyxl') as writer:
                    df_final_output.to_excel(writer, sheet_name='Final_Result', index=False)
                    if df_remove_logs is not None and not df_remove_logs.empty:
                        df_remove_logs.to_excel(writer, sheet_name='Removed_Logs', index=False)
                    if ai_logs:
                        pd.DataFrame(ai_logs).to_excel(writer, sheet_name='AI_Analysis_Logs', index=False)
                    for sheet_name, df_sheet in existing_sheets.items():
                        if sheet_name == 'Removed_Logs': continue
                        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                output_path = new_output_path
            except Exception as e2:
                print(f"保存新文件也失败: {e2}")

        except Exception as e:
            print(f"保存多 Sheet Excel 失败: {e}")
            df_final_output.to_excel(output_path, index=False)
    else:
        try:
            df_final_output.to_excel(output_path, index=False)
        except PermissionError:
            print(f"错误: 无法写入文件 {output_path}。文件可能被打开。")
            new_output_path = output_path.replace(".xlsx", f"_{int(time.time())}.xlsx")
            print(f"尝试保存到新文件: {new_output_path}")
            df_final_output.to_excel(new_output_path, index=False)
            output_path = new_output_path
        
    print(f"\nAI 分析完成！")
    print(f"原始条目数: {len(df)}")
    print(f"最终条目数: {len(df_final_output)}")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    run_ai_analysis()
