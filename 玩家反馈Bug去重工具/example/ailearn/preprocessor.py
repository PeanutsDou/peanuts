# -*- coding: utf-8 -*-
# @Author: Xuguoliang
# @Date:   2025-12-17 10:00:00
# @Version: 1.0.0

import re
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
from typing import List

class Preprocessor:
    """
    数据预处理模块：负责清洗、标准化和去重
    """

    def __init__(self, config):
        self.config = config

    def clean_text(self, text: str) -> str:
        """
        标准化清洗文本
        1. 转换为字符串
        2. 去除首尾空格
        3. 去除特殊不可见字符
        4. 统一标点符号 (可选)
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip()
        
        # 去除不可见字符 (如 \u200b 等)
        text = re.sub(r'[\u200b\u200e\u200f\ufeff]', '', text)
        
        # 替换多个空白为一个空格
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def is_valid(self, text: str) -> bool:
        """
        过滤无效信息
        1. 长度过短
        2. 纯数字/符号
        """
        if len(text) < self.config.MIN_TEXT_LENGTH:
            return False
            
        # 检查是否包含至少一个中文或英文字符
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z]', text):
            return False
            
        return True

    def _similarity(self, a: str, b: str) -> float:
        """计算文本相似度"""
        return SequenceMatcher(None, a, b).ratio()

    def remove_duplicates(self, df: pd.DataFrame, content_col: str) -> pd.DataFrame:
        """
        基于内容相似度去重
        注意：对于大数据量，O(N^2) 效率较低，此处仅做演示。
        生产环境建议使用 SimHash 或 Embedding 向量检索。
        """
        unique_indices = []
        texts = df[content_col].tolist()
        kept_texts = []
        
        print(f"正在进行去重处理，共 {len(texts)} 条数据...")
        
        # 使用 tqdm 显示进度
        for i in tqdm(range(len(texts)), desc="Deduplicating"):
            current_text = texts[i]
            
            # 如果是空或无效，跳过 (会被过滤掉)
            if not self.is_valid(current_text):
                continue
            
            is_duplicate = False
            
            # 简单优化：如果完全相同，直接跳过
            if current_text in kept_texts:
                continue

            # 相似度比对
            for kept in kept_texts:
                # 长度快速剪枝 (长度差异大则认为不相似)
                if abs(len(current_text) - len(kept)) / max(len(current_text), len(kept)) > 0.3:
                    continue
                    
                if self._similarity(current_text, kept) >= self.config.SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_texts.append(current_text)
                unique_indices.append(df.index[i])
                
        return df.loc[unique_indices].copy()
