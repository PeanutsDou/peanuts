# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json

# 尝试导入 sentence_transformers，如果不存在则提示
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence_transformers not found. Embedding step will fail.")

def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    """从 config.json 加载配置"""
    try:
        config_path = os.path.join(_script_dir(), "json", "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置失败，使用默认参数: {e}")
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
    """判断目录是否为 SentenceTransformer 本地模型目录（包含必要文件）"""
    if not path or not os.path.isdir(path):
        return False
    files = os.listdir(path)
    # 常见必备文件：config.json / modules.json
    needed = {"config.json", "modules.json"}
    return any(f in files for f in needed)

def _select_local_model_dir(cfg):
    """根据配置或约定位置选择本地模型目录"""
    # 1. 优先使用配置项 embedding_local_dir
    local_dir = cfg.get("embedding_local_dir") if isinstance(cfg, dict) else None
    if local_dir and _is_sentence_transformer_dir(local_dir):
        return local_dir
    # 2. 探测常见目录
    for cand in _candidate_local_model_dirs():
        if _is_sentence_transformer_dir(cand):
            return cand
    return None

def _ensure_dir(path):
    """确保目录存在"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _next_versioned_file(dir_path, base_name, ext):
    """生成带日期与版本号的文件名，如 embeddings_YYYYMMDD_vN.ext"""
    ts = pd.Timestamp.now().strftime('%Y%m%d')
    prefix = f"{base_name}_{ts}_v"
    existing = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(f".{ext}")]
    version = len(existing) + 1
    return os.path.join(dir_path, f"{base_name}_{ts}_v{version}.{ext}")

class TextEmbedder:
    """
    文本向量化处理类
    原理：使用预训练的深度学习模型（Sentence-Transformer）将文本转换为高维向量。
    向量捕捉了文本的语义信息，语义相似的文本在向量空间中距离更近。
    """
    def __init__(self, model_name=None):
        """
        初始化 Embedder
        :param model_name: 模型名称，若为None则尝试从配置读取，否则使用 moka-ai/m3e-base
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("请安装 sentence-transformers 库: pip install sentence-transformers")
        
        config = load_config()
        # 1) 先尝试本地目录（避免重复下载）
        local_dir = _select_local_model_dir(config)
        if local_dir:
            print(f"检测到本地模型目录: {local_dir}，优先使用本地模型。")
            self.model = SentenceTransformer(local_dir)
        else:
            # 2) 回退到配置中的模型名称或默认 m3e-base
            if model_name is None:
                model_name = config.get("embedding_model", "moka-ai/m3e-base")
            print(f"正在加载 Embedding 模型: {model_name} ...")
            print("如果是首次使用该模型，会自动从 HuggingFace 下载，请保持网络通畅。")
            # 如果下载慢，可以考虑使用本地模型路径，或更换为国内源
            self.model = SentenceTransformer(model_name)
        print("模型加载完成。")

    def encode(self, texts):
        """
        将文本列表转换为向量列表
        :param texts: 文本字符串列表
        :return: numpy array 格式的向量矩阵
        """
        print(f"正在将 {len(texts)} 条文本转换为向量...")
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings

def run_embedding(df, column='subject'):
    """
    执行向量化流程
    :param df: 包含文本数据的 DataFrame
    :param column: 要进行向量化的列名
    :return: (df_emb, emb_path)
             - df_emb: 原数据的副本，新增 'embedding_index' 列（向量矩阵的索引）
             - emb_path: 向量矩阵保存的 .npy 文件路径
    """
    if df is None or df.empty:
        print("没有数据需要向量化。")
        return df, None

    if column not in df.columns:
        print(f"Error: 列 '{column}' 不存在。")
        return df, None

    try:
        # 初始化时不传 model_name，让其自动读取配置
        embedder = TextEmbedder()
        texts = df[column].astype(str).tolist()
        embeddings = embedder.encode(texts)
        
        # 读取存储配置
        cfg = load_config()
        store_dir = os.path.join(_script_dir(), cfg.get("embedding_dir", "embeddings"))
        _ensure_dir(store_dir)
        
        # 方案B：高性能保存为 .npy 文件，并在 DataFrame 内仅保存索引
        emb_path = _next_versioned_file(store_dir, "embeddings", "npy")
        np.save(emb_path, np.array(embeddings, dtype=np.float32))
        
        # 返回索引映射，不在 DataFrame 内保存大向量
        df_emb = df.copy()
        df_emb['embedding_index'] = list(range(len(texts)))
        
        print(f"向量化完成，已保存为 .npy: {emb_path}")
        return df_emb, emb_path
        
    except Exception as e:
        print(f"向量化过程中发生错误: {e}")
        return df, None
