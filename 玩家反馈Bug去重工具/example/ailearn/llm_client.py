# -*- coding: utf-8 -*-
# @Author: Xuguoliang
# @Date:   2025-12-23 11:00:00
# @Version: 1.1.1

import json
import requests
import time
import logging
import sqlite3
import hashlib
import os
from typing import List, Dict, Optional, Generator, Union, Any
from dataclasses import dataclass
from threading import Lock

@dataclass
class LLMConfig:
    """
    LLM 配置结构体
    """
    app_id: str
    app_key: str
    base_url: str
    model: str
    embedding_model: str = ""
    max_tokens: int = 2000 # 增加 token 上限以适应批处理
    temperature: float = 0.01 # 温度参数，越低越确信 (0.0-1.0)
    timeout: int = 120
    max_retries: int = 5        # 最大重试次数
    retry_delay: float = 2.0    # 初始重试延迟（秒）
    max_retry_delay: float = 60.0 # 最大重试延迟（秒）
    cache_file: str = "llm_cache.db" # 缓存文件路径

class SQLiteCache:
    """
    简单的 SQLite 缓存实现
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = Lock()
        self._init_db()

    def _init_db(self):
        # 确保目录存在
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS llm_cache (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        timestamp REAL
                    )
                ''')
                conn.commit()
                conn.close()
            except Exception as e:
                logging.error(f"Failed to init cache db: {e}")

    def get(self, key: str) -> Optional[str]:
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM llm_cache WHERE key = ?', (key,))
                row = cursor.fetchone()
                conn.close()
                return row[0] if row else None
            except Exception as e:
                logging.error(f"Cache get error: {e}")
                return None

    def set(self, key: str, value: str):
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('REPLACE INTO llm_cache (key, value, timestamp) VALUES (?, ?, ?)', 
                             (key, value, time.time()))
                conn.commit()
                conn.close()
            except Exception as e:
                logging.error(f"Cache set error: {e}")

class LLMClient:
    """
    LLM 客户端，用于与大语言模型服务进行交互
    支持对话（Chat）和文本嵌入（Embedding）
    支持自动重试机制
    支持本地缓存
    """

    def __init__(self, config: LLMConfig):
        """
        初始化 LLM 客户端

        Args:
            config (LLMConfig): 客户端配置信息
        """
        self.config = config
        self.session = requests.Session()
        # 设置鉴权 Header
        if self.config.app_id:
            api_key = f"{self.config.app_id}.{self.config.app_key}"
        else:
            api_key = self.config.app_key
            
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        # 处理 Base URL，移除末尾的斜杠
        self.base_url = self.config.base_url.strip().rstrip("/")
        
        # 配置日志
        self.logger = logging.getLogger("LLMClient")
        
        # 初始化缓存
        self.cache = SQLiteCache(self.config.cache_file)

    def _generate_cache_key(self, model: str, messages: Any) -> str:
        """生成缓存 Key"""
        content = json.dumps({
            "model": model,
            "messages": messages
        }, sort_keys=True)
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        带重试机制的请求封装
        """
        retries = 0
        current_delay = self.config.retry_delay
        
        while retries <= self.config.max_retries:
            try:
                response = self.session.request(method, url, **kwargs)
                
                # 如果是 429 Too Many Requests 或 5xx Server Error，则重试
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    retries += 1
                    if retries > self.config.max_retries:
                        self.logger.error(f"Max retries exceeded. Last status: {response.status_code}")
                        response.raise_for_status()
                        return response
                        
                    # 指数退避策略
                    self.logger.warning(f"Request failed with status {response.status_code}. Retrying in {current_delay:.2f}s ({retries}/{self.config.max_retries})...")
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 2, self.config.max_retry_delay)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > self.config.max_retries:
                    raise e
                    
                self.logger.warning(f"Request exception: {str(e)}. Retrying in {current_delay:.2f}s ({retries}/{self.config.max_retries})...")
                time.sleep(current_delay)
                current_delay = min(current_delay * 2, self.config.max_retry_delay)

    def create_embedding(self, input_text: str) -> List[float]:
        """
        创建文本嵌入 (不做缓存，通常不需要)
        """
        url = f"{self.base_url}/embeddings"
        
        model = self.config.embedding_model
        if not model:
            model = self.config.model

        payload = {
            "model": model,
            "input": input_text
        }

        try:
            response = self._request_with_retry(
                "POST",
                url, 
                json=payload, 
                timeout=self.config.timeout
            )
            
            data = response.json()
            if not data.get("data"):
                raise ValueError("No embedding data in response")
            
            return data["data"][0]["embedding"]

        except Exception as e:
            raise Exception(f"Create embedding failed: {str(e)}")

    def chat(self, messages: Union[List[Dict[str, str]], str]) -> str:
        """
        发起非流式对话 (带缓存)
        """
        # 如果传入的是字符串，自动封装为消息列表
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # 1. 检查缓存
        cache_key = self._generate_cache_key(self.config.model, messages)
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return cached_response

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

        try:
            response = self._request_with_retry(
                "POST",
                url, 
                json=payload, 
                timeout=self.config.timeout
            )
            
            data = response.json()
            if not data.get("choices"):
                raise ValueError("No choices in chat response")
                
            content = data["choices"][0]["message"]["content"]
            
            # 2. 写入缓存
            self.cache.set(cache_key, content)
            
            return content

        except Exception as e:
            raise Exception(f"Chat request failed: {str(e)}")
