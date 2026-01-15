# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os
import json
import time

def _script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    """从 config.json 加载配置"""
    try:
        config_path = os.path.join(_script_dir(), "json", "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置失败: {e}")
        return {}

def run_kmeans_partition(df, emb_path_or_array, n_clusters=10):
    """
    执行 K-Means 粗略分区
    :param df: 数据 DataFrame (需包含 embedding_index 或其他标识)
    :param emb_path_or_array: 向量文件路径 或 numpy array
    :param n_clusters: 目标簇数量 (批次数)
    :return: list of (batch_id, batch_df, batch_embeddings)
    """
    start_time = time.time()
    
    # 1. 加载向量
    embeddings = None
    if isinstance(emb_path_or_array, str):
        if os.path.isfile(emb_path_or_array):
            embeddings = np.load(emb_path_or_array)
        else:
            print(f"错误: 向量文件未找到: {emb_path_or_array}")
            return []
    else:
        embeddings = emb_path_or_array
        
    if embeddings is None or len(embeddings) == 0:
        print("K-Means: 无有效向量数据。")
        return []
        
    # 确保 n_clusters 不超过数据量
    n_samples = len(embeddings)
    if n_clusters > n_samples:
        n_clusters = max(1, n_samples // 2)
        print(f"警告: 数据量 ({n_samples}) 小于 目标簇数，自动调整 n_clusters = {n_clusters}")
    
    print(f"开始 K-Means 粗略分区 (目标: {n_clusters} 个批次, 总数据量: {n_samples})...")
    
    # 2. 执行 K-Means
    try:
        # random_state 固定以保证可复现
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # 3. 拆分数据
        batches = []
        # labels 是一个与 embeddings 等长的数组
        
        # 为了高效拆分，我们先将 labels 附加到 df (如果是临时列)
        # 注意: 这里假设 df 的顺序与 embeddings 一致
        if len(df) != len(labels):
            print(f"严重错误: DataFrame 长度 ({len(df)}) 与 向量长度 ({len(labels)}) 不一致!")
            return []
            
        # 统计各批次大小
        from collections import Counter
        counts = Counter(labels)
        print(f"K-Means 分区完成，各批次分布: {dict(counts)}")
        
        # 拆分
        # 获取 df 的索引，方便切片
        # 我们直接根据 label 索引来切分 embeddings
        
        for i in range(n_clusters):
            if i not in counts:
                continue
                
            # 获取该批次的索引掩码
            mask = (labels == i)
            
            # 切分 DataFrame
            batch_df = df[mask].copy()
            
            # 切分 Embeddings
            batch_embeddings = embeddings[mask]
            
            batches.append((i, batch_df, batch_embeddings))
            
        end_time = time.time()
        print(f"K-Means 分区耗时: {end_time - start_time:.2f}秒")
        
        return batches
        
    except Exception as e:
        print(f"K-Means 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # 测试代码
    pass
