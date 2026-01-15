# -*- coding: utf-8 -*-
import os
import json
import time
import requests
import re
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def _script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    """加载配置文件"""
    cfg_path = os.path.join(_script_dir(), "json", "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_embeddings(embeddings):
    """
    L2 归一化向量
    """
    X = np.array(list(embeddings), dtype=np.float32)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    return X / norm

def call_deepseek_review(subjects, config):
    """
    调用 DeepSeek 对聚类代表进行分析，找出需要合并的组
    subjects: [(cluster_id, representative_text), ...]
    返回: 合并映射 {old_cluster_id: new_cluster_id}
    """
    llm_cfg = config.get("llm_settings", {})
    app_id = llm_cfg.get("app_id", "")
    app_key = llm_cfg.get("app_key", "")
    base_url = llm_cfg.get("base_url", "").rstrip("/")
    # 使用配置中指定的 review_model，如果未配置则回退到 default model 或 deepseek-v3.2-latest
    model = llm_cfg.get("review_model", llm_cfg.get("model", "deepseek-v3.2-latest"))
    
    if not app_key:
        print("错误: LLM app_key 未配置，跳过 AI 复查。")
        return {}
        
    auth_key = f"{app_id}.{app_key}" if app_id else app_key
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }
    
    # 构造 Prompt
    # 将 cluster_id 和 文本 结合
    items_text = "\n".join([f"ID:{cid} 内容:{text[:200]}" for cid, text in subjects])
    
    prompt = f"""
你是一位资深游戏QA专家。以下是一组聚类后的Bug/建议的代表性描述。
请分析这些描述的语义，找出表达意思完全相同或高度重合的组。
有些描述可能用词不同，但指的是同一个游戏机制问题，这些也应该合并。

输入列表：
{items_text}

请返回一个 JSON 列表，列表中的每个元素是一个包含需要合并的 ID 的数组。
不需要合并的 ID 不用返回。
例如，如果 ID:1 和 ID:5 是同一个问题，ID:3 和 ID:8 是同一个问题，请返回：
[[1, 5], [3, 8]]

只返回 JSON 数组，不要其他解释。
"""
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        
        # 清理 Markdown
        clean_content = re.sub(r'^```json\s*', '', content.strip())
        clean_content = re.sub(r'^```\s*', '', clean_content)
        clean_content = re.sub(r'\s*```$', '', clean_content)
        
        merged_groups = json.loads(clean_content)
        
        mapping = {}
        if isinstance(merged_groups, list):
            for group in merged_groups:
                if isinstance(group, list) and len(group) > 1:
                    # 将同组的 ID 映射到该组第一个 ID
                    target_id = group[0]
                    for source_id in group[1:]:
                        mapping[source_id] = target_id
        return mapping
        
    except Exception as e:
        print(f"DeepSeek 复查调用失败: {e}")
        return {}

def merge_clusters_with_llm(df, config):
    """
    步骤 1: 使用 LLM 检查过度聚类
    """
    print(">>> 开始 AI 聚类复查 (合并过度聚类)...")
    if 'cluster_id' not in df.columns or df.empty:
        return df, []
        
    # 提取每个簇的代表性条目
    # 忽略噪点 -1
    unique_clusters = [c for c in df['cluster_id'].unique() if c != -1]
    if not unique_clusters:
        return df, []
        
    representatives = []
    for cid in unique_clusters:
        cluster_df = df[df['cluster_id'] == cid]
        # 简单取最长作为代表
        subjects = cluster_df['subject'].astype(str).tolist()
        if subjects:
            rep_text = max(subjects, key=len)
            representatives.append((cid, rep_text))
            
    # 如果簇太多，可能需要分批处理，这里暂不分批（假设 DeepSeek 能处理长 Context）
    # 但为了稳妥，限制每次处理 100 个簇
    batch_size = 100
    total_mapping = {}
    
    for i in range(0, len(representatives), batch_size):
        batch = representatives[i:i+batch_size]
        print(f"    正在分析第 {i+1}-{i+len(batch)} 个簇代表...")
        mapping = call_deepseek_review(batch, config)
        total_mapping.update(mapping)
        
    if not total_mapping:
        print("    未发现需要合并的聚类。")
        return df, []
        
    print(f"    发现 {len(total_mapping)} 个簇需要合并。")
    
    # 应用合并
    # 为了记录日志，我们先拷贝一份
    df_new = df.copy()
    logs = []
    
    for old_id, new_id in total_mapping.items():
        # 更新 cluster_id
        mask = df_new['cluster_id'] == old_id
        count = mask.sum()
        df_new.loc[mask, 'cluster_id'] = new_id
        
        logs.append({
            "action": "merge_cluster",
            "old_cluster_id": old_id,
            "new_cluster_id": new_id,
            "count": count
        })
        
    return df_new, logs

def recover_noise_with_dbscan(df, embeddings, config):
    """
    步骤 2: 对噪点进行 DBSCAN 二次聚类
    """
    print(">>> 开始噪点回收 (DBSCAN)...")
    if 'cluster_id' not in df.columns:
        return df, []
        
    noise_mask = df['cluster_id'] == -1
    noise_df = df[noise_mask]
    
    if noise_df.empty:
        print("    没有噪点需要处理。")
        return df, []
        
    # 获取噪点对应的向量
    # 注意：embeddings 的索引必须与 df 的索引对齐
    # 假设 df 是经过 reset_index 的，或者我们通过 index 匹配
    # 为了安全，我们最好根据 index 获取
    try:
        noise_indices = noise_df.index
        # 确保 indices 在 embeddings 范围内
        if max(noise_indices) >= len(embeddings):
            # 如果索引不匹配（可能经过了重置），尝试按位置
            # 这通常发生在 df 被切片过。
            # 最稳妥的方式是如果在 run_review 里传入的是完整的 embeddings 和完整的 df
            # 如果不是，我们需要在 df 里存 embedding index 或者直接存 embedding
            # 这里的 df 通常来自 dpdata.run_clustering，索引应该是连续的或者保留了原始索引
            # 让我们尝试直接用 boolean mask
            noise_vectors = embeddings[noise_mask]
        else:
            noise_vectors = embeddings[noise_indices]
    except Exception as e:
        print(f"    无法获取噪点向量: {e}")
        return df, []
        
    if len(noise_vectors) < 2:
        return df, []
        
    # 归一化
    X_noise = normalize_embeddings(noise_vectors)
    
    # DBSCAN 参数：比 HDBSCAN 稍宽松
    # 假设 HDBSCAN 的 epsilon 映射是 0.1 (配置里的 cluster_selection_epsilon)
    # 我们这里可以稍微大一点点，或者保持一致但 min_samples 设为 2
    settings = config.get("dbscan_settings", {}) # 这里读取的是原本用于 HDBSCAN 的配置
    # 注意：在 config.json 里 dbscan_settings 的 eps 默认是 0.05 (较小)
    # 而我们刚才在 DBSCAN_data.py 里把 eps 映射为了 0.1 给 HDBSCAN
    # 这里我们给一个稍微宽松的值，比如 0.15 或 0.2
    # 既然用户说“比主聚类流程宽松一点点”，如果主流程 eps 是 0.1，这里可以用 0.15
    eps = 0.15 
    min_samples = 2
    
    print(f"    DBSCAN 参数: eps={eps}, min_samples={min_samples}, 噪点数量={len(noise_df)}")
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X_noise)
    
    # 分配新的 Cluster ID
    # 新 ID 应该比现有的最大 ID 大
    max_id = df['cluster_id'].max()
    if max_id == -1: max_id = 0
    
    new_labels = labels.copy()
    # 只有 label != -1 的才分配新 ID
    # 加上 (max_id + 1)
    # 注意：labels 里的 0, 1, 2... 要变成 max_id+1, max_id+2...
    valid_mask = labels != -1
    new_labels[valid_mask] = labels[valid_mask] + (max_id + 1)
    
    # 更新 df
    # 注意要用 loc 更新
    # noise_df 是切片，不能直接更新回去，要操作原 df
    # 我们需要构建一个与 noise_df 长度相同的数组，对应 noise_df 的每行
    
    # 统计回收情况
    recovered_count = np.sum(valid_mask)
    n_new_clusters = len(set(new_labels[valid_mask])) if recovered_count > 0 else 0
    
    print(f"    从噪点中找回 {recovered_count} 条目，归入 {n_new_clusters} 个新簇。")
    
    if recovered_count > 0:
        # 更新原 DataFrame
        # df.loc[noise_mask, 'cluster_id'] = new_labels # 这样写可能不行，因为 new_labels 包含 -1
        # 我们只更新那些被 DBSCAN 聚类的点
        
        # 找到那些被聚类的数据在原 df 中的索引
        # noise_df.index 是原 df 的索引
        recovered_indices = noise_df.index[valid_mask]
        recovered_new_ids = new_labels[valid_mask]
        
        df.loc[recovered_indices, 'cluster_id'] = recovered_new_ids
        
    return df, [{"action": "recover_noise", "recovered_count": int(recovered_count), "new_clusters": int(n_new_clusters)}]

def run_review(df, emb_path):
    """
    主入口
    """
    print("\n=== 步骤 2.5: 聚类结果复查与优化 ===")
    config = load_config()
    
    # 加载向量
    if not os.path.exists(emb_path):
        print("未找到向量文件，跳过复查。")
        return df, []
        
    embeddings = np.load(emb_path)
    if len(embeddings) != len(df):
        print(f"警告: 向量数量 ({len(embeddings)}) 与 数据量 ({len(df)}) 不一致，跳过复查。")
        return df, []
        
    logs = []
    
    # 1. LLM 合并过度聚类
    try:
        df, merge_logs = merge_clusters_with_llm(df, config)
        logs.extend(merge_logs)
    except Exception as e:
        print(f"LLM 合并步骤异常: {e}")
        
    # 2. DBSCAN 找回噪点
    try:
        df, recover_logs = recover_noise_with_dbscan(df, embeddings, config)
        logs.extend(recover_logs)
    except Exception as e:
        print(f"DBSCAN 找回步骤异常: {e}")
        
    # 3. 重新计算内聚度 (针对变动的簇)
    # 为了简单，我们可以重新计算所有非噪点簇的内聚度
    # 复用代码逻辑
    print("    更新内聚度指标...")
    X_norm = normalize_embeddings(embeddings)
    
    unique_ids = df['cluster_id'].unique()
    for cid in unique_ids:
        if cid == -1: continue
        mask = df['cluster_id'] == cid
        if mask.sum() < 2:
            # 单点簇内聚度设为 1.0 (或保持原样)
            # df.loc[mask, 'cohesion'] = 1.0
            continue
            
        cluster_vectors = X_norm[mask]
        centroid = np.mean(cluster_vectors, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        
        similarities = np.dot(cluster_vectors, centroid)
        cohesion = float(np.mean(similarities))
        df.loc[mask, 'cohesion'] = cohesion
        
    return df, logs
