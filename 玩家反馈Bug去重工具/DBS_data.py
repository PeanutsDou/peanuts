# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
import warnings
import time

# 尝试导入 HDBSCAN
try:
    from sklearn.cluster import HDBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("警告: 未安装 scikit-learn，聚类功能将不可用。")

# 忽略 FutureWarnings
warnings.filterwarnings("ignore")

def _script_dir():
    """获取脚本所在目录"""
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
            
            # 兼容旧配置
            old_settings = config.get("dbscan_settings", {})
            # 这里的 eps 原本用于 DBSCAN，现在映射给 HDBSCAN 的 cluster_selection_epsilon
            # cluster_selection_epsilon > 0 会让 HDBSCAN 表现得更像 DBSCAN，合并距离小于该值的簇
            return {
                "min_cluster_size": old_settings.get("min_samples", 2),
                "min_samples": 1, # 显式设置为 1，保留更多核心点，避免过多噪点
                "cluster_selection_epsilon": old_settings.get("eps", 0.1), # 默认给一个非 0 值，允许一定程度的合并
                "metric": "euclidean"
            }
    except Exception as e:
        print(f"加载聚类配置失败: {e}")
        return {"min_cluster_size": 2, "min_samples": 1, "cluster_selection_epsilon": 0.1, "metric": "euclidean"}

def normalize_embeddings(embeddings):
    """
    L2 归一化向量
    目的：归一化后，欧氏距离等价于余弦距离
    注意：HDBSCAN 使用 euclidean 距离，对归一化向量效果最好
    """
    X = np.array(list(embeddings), dtype=np.float32)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10 # 避免除零
    return X / norm

def run_clustering(df, emb_path=None):
    """
    执行 HDBSCAN 聚类
    :param df: 数据 DataFrame
    :param emb_path: 向量文件路径
    :return: (df_result, cluster_logs)
    """
    start_time = time.time()
    
    if df is None or df.empty:
        return df, []
    
    if not HAS_SKLEARN:
        return df, []

    settings = load_cluster_config()
    min_cluster_size = settings.get("min_cluster_size", 2)
    min_samples = settings.get("min_samples", 1)
    cluster_selection_epsilon = settings.get("cluster_selection_epsilon", 0.1)
    
    if min_cluster_size < 2: min_cluster_size = 2

    print(f"HDBSCAN 参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={cluster_selection_epsilon}")

    try:
        # 加载向量
        embeddings = None
        if emb_path and os.path.isfile(emb_path):
            embeddings = np.load(emb_path)
        elif 'embedding' in df.columns:
            embeddings = np.array(list(df['embedding']), dtype=np.float32)
        elif 'embedding_index' in df.columns:
            print("错误: 有 embedding_index 但无 emb_path。")
            return df, []
        
        if embeddings is None or len(embeddings) == 0:
            print("没有可用向量数据。")
            return df, []
            
        if len(embeddings) < 2:
            print(f"数据量 ({len(embeddings)}) 少于2，无法进行 HDBSCAN 聚类，全部标记为噪点。")
            df_result = df.copy()
            df_result['cluster_id'] = -1
            df_result['cohesion'] = 0.0
            return df_result, []

        print(f"开始聚类 {len(embeddings)} 条数据...")

        # 1. 归一化
        X_norm = normalize_embeddings(embeddings)
        
        # 2. 聚类
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X_norm)
        
        # 3. 结果整理
        df_result = df.copy()
        df_result['cluster_id'] = labels
        df_result['cohesion'] = 0.0

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"聚类完成: {n_clusters} 个簇, {n_noise} 个噪点。")

        # 计算内聚度 (Cohesion)
        cluster_logs = []
        
        # 尝试引入 tqdm
        try:
            from tqdm import tqdm
            iter_labels = tqdm(unique_labels, desc="计算簇内聚度")
        except ImportError:
            iter_labels = unique_labels
        
        for label in iter_labels:
            if label == -1: continue

            mask = (labels == label)
            cluster_vectors = X_norm[mask]
            
            # 计算质心
            centroid = np.mean(cluster_vectors, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            
            # 计算平均相似度
            similarities = np.dot(cluster_vectors, centroid)
            cohesion = float(np.mean(similarities))
            
            # 写入 df
            # 注意：假设 df 行顺序与 embeddings 一致
            df_result.loc[mask, 'cohesion'] = cohesion

            # 日志
            cluster_items = df_result[mask]
            subjects = cluster_items['subject'].tolist()
            # 选取最长作为代表
            representative = max(subjects, key=len) if subjects else ""

            cluster_logs.append({
                "Cluster ID": int(label),
                "Count": len(subjects),
                "Cohesion": round(cohesion, 4),
                "Representative": representative,
                "Subjects": " | ".join(subjects[:5]) + ("..." if len(subjects)>5 else "")
            })

        cluster_logs.sort(key=lambda x: x["Count"], reverse=True)
        
        end_time = time.time()
        print(f"聚类耗时: {end_time - start_time:.2f}秒")
        return df_result, cluster_logs

    except Exception as e:
        print(f"聚类错误: {e}")
        import traceback
        traceback.print_exc()
        return df, []
