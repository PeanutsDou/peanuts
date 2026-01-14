# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
import os
import json
import warnings

# 忽略可能的 FutureWarnings
warnings.filterwarnings("ignore")

def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_cluster_config():
    """从 config.json 加载聚类配置"""
    try:
        config_path = os.path.join(_script_dir(), "json", "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
            # 优先查找 hdbscan_settings
            if "hdbscan_settings" in config:
                return config["hdbscan_settings"]
            
            # 兼容旧配置，提取有用参数
            old_settings = config.get("dbscan_settings", {})
            return {
                "min_cluster_size": old_settings.get("min_samples", 2), # 复用 min_samples 作为最小簇大小
                "min_samples": None, # 让 HDBSCAN 自动处理
                "cluster_selection_epsilon": 0.0,
                "metric": "euclidean"
            }
    except Exception as e:
        print(f"加载配置失败，使用默认参数: {e}")
        return {"min_cluster_size": 2, "metric": "euclidean"}

def normalize_embeddings(embeddings):
    """
    对向量进行 L2 归一化，以便使用欧氏距离近似余弦距离
    """
    X = np.array(list(embeddings), dtype=np.float32)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10 # 避免除零
    return X / norm

def run_clustering(df, emb_path=None):
    """
    执行 HDBSCAN 聚类流程
    :param df: 包含 'embedding' 列的 DataFrame
    :param emb_path: 向量矩阵 .npy 文件路径（优先使用该路径加载向量）
    :return: 
        1. 带有 'cluster_id' 的 DataFrame
        2. 聚类日志列表（用于生成日志表）
    """
    if df is None or df.empty:
        print("没有可用的数据进行聚类。")
        return df, []

    settings = load_cluster_config()
    min_cluster_size = settings.get("min_cluster_size", 2)
    min_samples = settings.get("min_samples", None)
    cluster_selection_epsilon = settings.get("cluster_selection_epsilon", 0.0)
    
    # 确保 min_cluster_size 至少为 2
    if min_cluster_size < 2:
        min_cluster_size = 2

    print(f"正在配置 HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    try:
        # 加载向量矩阵
        if emb_path and os.path.isfile(emb_path):
            embeddings = np.load(emb_path)
        elif 'embedding' in df.columns:
            embeddings = np.array(list(df['embedding']), dtype=np.float32)
        elif 'embedding_index' in df.columns:
            print("检测到 embedding_index，但未提供 emb_path，无法加载向量矩阵。")
            return df, []
        else:
            print("未找到向量数据列或文件路径，无法进行聚类。")
            return df, []

        n_samples = len(embeddings)
        if n_samples == 0:
            print("没有可用的向量进行聚类。")
            return df, []
            
        print(f"加载向量数据完成，共 {n_samples} 条，开始执行 HDBSCAN 聚类...")

        # 1. 归一化向量 (Cosine -> Euclidean)
        X_norm = normalize_embeddings(embeddings)
        
        # 2. 初始化 HDBSCAN
        # metric='euclidean' 在 L2 归一化后等价于 Cosine Distance 排序
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom' # 'eom' (Excess of Mass) 通常更稳健
        )
        
        # 3. 拟合与预测
        labels = clusterer.fit_predict(X_norm)
        
        # 4. 结果处理
        df_result = df.copy()
        df_result['cluster_id'] = labels
        # 初始化 cohesion 列，默认 0.0 (float)
        df_result['cohesion'] = 0.0

        cluster_logs = []
        unique_labels = set(labels)

        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / n_samples if n_samples > 0 else 0
        
        print(f"聚类完成: 发现 {n_clusters} 个聚类，{n_noise} 个离群点 (占比 {noise_ratio:.2%})。")

        for label in unique_labels:
            if label == -1:
                continue

            # 获取该簇的索引和向量
            mask = (labels == label)
            cluster_indices = np.where(mask)[0]
            cluster_vectors = X_norm[cluster_indices] # 已经是归一化的
            
            # 计算 Cohesion (平均余弦相似度)
            # 1. 计算质心 (归一化后的向量平均，再归一化)
            centroid = np.mean(cluster_vectors, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
                
            # 2. 计算每个点到质心的余弦相似度 (dot product)
            # 因为都已经归一化，dot product 就是 cosine similarity
            similarities = np.dot(cluster_vectors, centroid)
            cohesion = float(np.mean(similarities))
            
            # 将 cohesion 写入 df_result
            # 注意：这里的 cluster_indices 是基于 embeddings 数组的下标
            # 假设 df 的 index 没有被打乱且与 embeddings 一一对应
            # 如果 df 是经过过滤的，我们需要确保 df_result 的 iloc 与 embeddings 下标对应
            # run_clustering 接收的 df 通常是经过 semantic_deduplicate 后的，index 可能不连续
            # 但 embeddings 是 list(df['embedding']) 生成的，所以顺序与 df 目前的行顺序一致
            # 因此使用 iloc 赋值是安全的
            df_result.iloc[cluster_indices, df_result.columns.get_loc('cohesion')] = cohesion

            cluster_items = df_result[df_result['cluster_id'] == label]
            subjects = cluster_items['subject'].tolist()
            ids = cluster_items.index.tolist()

            representative = max(subjects, key=len) if subjects else ""

            cluster_logs.append({
                "Cluster ID": int(label),
                "Count": len(subjects),
                "Cohesion": round(cohesion, 4),
                "Representative": representative,
                "Subjects": " | ".join(subjects),
                "IDs": str(ids)
            })

        # 按簇大小降序排列
        cluster_logs.sort(key=lambda x: x["Count"], reverse=True)

        return df_result, cluster_logs

    except Exception as e:
        print(f"聚类过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return df, []

if __name__ == "__main__":
    # 简单测试
    print("Testing HDBSCAN module...")
    data = [
        "Group A - 1", "Group A - 2", "Group A - 3", 
        "Group B - 1", "Group B - 2", 
        "Noise"
    ]
    # 模拟 Embedding (3维)
    # A 组: 接近 [0, 0, 1]
    # B 组: 接近 [1, 0, 0]
    # Noise: [0, 1, 0]
    embs = [
        [0.01, 0.01, 0.99],
        [0.02, 0.01, 0.98],
        [0.01, 0.02, 0.98],
        [0.99, 0.01, 0.01],
        [0.98, 0.02, 0.01],
        [0.01, 0.99, 0.01]
    ]
    df = pd.DataFrame({"subject": data, "embedding": embs})
    
    # 临时写入 config (Simulated)
    # Note: load_cluster_config will read from disk, so we rely on defaults or what's in config.json
    # Ensure min_cluster_size=2 for this small test
    
    df_res, logs = run_clustering(df)
    print("Cluster IDs:", df_res['cluster_id'].tolist())
    print("Cohesions:", df_res['cohesion'].tolist())
    print("Logs:", logs)
