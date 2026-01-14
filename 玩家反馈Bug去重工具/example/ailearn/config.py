# -*- coding: utf-8 -*-
# @Author: Xuguoliang
# @Date:   2025-12-17 10:00:00
# @Version: 1.0.0

import os

class Config:
    """
    全局配置类
    """
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = None # 需通过命令行参数传入，例如: python main.py data.xlsx
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 输出文件
    import pandas as pd
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"Feedback_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    # Excel 列名映射
    COL_CONTENT = "主题"  
    
    # 清洗配置
    SIMILARITY_THRESHOLD = 0.85  # 相似度阈值 (0-1)，越高越严格
    MIN_TEXT_LENGTH = 1          # 最短有效文本长度
    
    # LLM 配置 (网易 AI 网关)
    LLM_APP_ID = "fob3ubx6yv5jxjag"
    LLM_APP_KEY = "x55ll082jpilelrdvv57lt1ocoao726v"
    LLM_BASE_URL = "https://aigw.nie.netease.com/v1" 
    LLM_MODEL = "deepseek-v3.2-latest"
    
    # 线程池配置 (根据 API 限制调整)
    MAX_WORKERS = 20 # 配合重试机制适当提高并发
    
    # 批处理配置
    BATCH_SIZE = 20  # 每次请求处理的反馈数量
    
    # 缓存配置
    LLM_CACHE_FILE = "llm_cache.db"
