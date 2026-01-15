# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
import time

# 尝试导入 sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("警告: 未安装 sentence_transformers，向量化功能将不可用。")

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

def _candidate_local_model_dirs():
    """返回可能的本地模型目录列表（优先匹配用户路径）"""
    home = os.path.expanduser('~')
    candidates = [
        os.path.join('C:\\Users', 'douzhongjun', 'm3e-base'),
        os.path.join('C:\\Users', 'douzhongjun', 'moka-ai', 'm3e-base'),
        os.path.join(home, 'm3e-base'),
        os.path.join(home, 'moka-ai', 'm3e-base'),
    ]
    return candidates

def _is_sentence_transformer_dir(path):
    """判断目录是否为 SentenceTransformer 本地模型目录"""
    if not path or not os.path.isdir(path):
        return False
    files = os.listdir(path)
    needed = {"config.json", "modules.json"}
    return any(f in files for f in needed)

def _select_local_model_dir(cfg):
    """根据配置或约定位置选择本地模型目录"""
    local_dir = cfg.get("embedding_local_dir") if isinstance(cfg, dict) else None
    if local_dir and _is_sentence_transformer_dir(local_dir):
        return local_dir
    for cand in _candidate_local_model_dirs():
        if _is_sentence_transformer_dir(cand):
            return cand
    return None

def _ensure_dir(path):
    """确保目录存在"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _next_versioned_file(dir_path, base_name, ext):
    """生成带日期与版本号的文件名"""
    ts = pd.Timestamp.now().strftime('%Y%m%d')
    prefix = f"{base_name}_{ts}_v"
    existing = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(f".{ext}")]
    version = len(existing) + 1
    return os.path.join(dir_path, f"{base_name}_{ts}_v{version}.{ext}")

class TextEmbedder:
    """
    文本向量化处理类
    功能：将文本转换为高维向量
    """
    def __init__(self, model_name=None):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("请安装 sentence-transformers 库")
        
        config = load_config()
        # 优先使用本地模型
        local_dir = _select_local_model_dir(config)
        if local_dir:
            print(f"加载本地模型: {local_dir}")
            self.model = SentenceTransformer(local_dir)
        else:
            if model_name is None:
                model_name = config.get("embedding_model", "moka-ai/m3e-base")
            print(f"加载在线模型: {model_name}")
            self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        批量编码文本
        :param texts: 文本列表
        :return: 向量矩阵 (numpy array)
        """
        # normalize_embeddings=True 启用 L2 归一化
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings

def run_embedding(df, column='subject'):
    """
    执行向量化流程
    :param df: 数据 DataFrame
    :param column: 默认文本列名
    :return: (df_emb, emb_path)
    """
    start_time = time.time()
    if df is None or df.empty:
        return df, None

    # 优先使用 normalized_subject
    target_column = column
    if 'normalized_subject' in df.columns:
        print("检测到 normalized_subject，使用规范化文本进行向量化...")
        target_column = 'normalized_subject'
    elif column not in df.columns:
        print(f"错误: 列 '{column}' 不存在。")
        return df, None

    try:
        embedder = TextEmbedder()
        texts = df[target_column].astype(str).tolist()
        print(f"正在向量化 {len(texts)} 条文本...")
        embeddings = embedder.encode(texts)
        
        cfg = load_config()
        store_dir = os.path.join(_script_dir(), cfg.get("embedding_dir", "embeddings"))
        _ensure_dir(store_dir)
        
        # 保存为 .npy
        emb_path = _next_versioned_file(store_dir, "embeddings", "npy")
        np.save(emb_path, np.array(embeddings, dtype=np.float32))
        
        # DataFrame 仅记录索引，避免内存过大
        df_emb = df.copy()
        df_emb['embedding_index'] = list(range(len(texts)))
        
        end_time = time.time()
        print(f"向量化完成，已保存至: {emb_path}，耗时: {end_time - start_time:.2f}秒")
        return df_emb, emb_path
        
    except Exception as e:
        print(f"向量化错误: {e}")
        return df, None
