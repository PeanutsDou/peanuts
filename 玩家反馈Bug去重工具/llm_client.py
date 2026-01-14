# -*- coding: utf-8 -*-
import requests
import json
import os
import time
import logging
import sqlite3
import hashlib
from threading import Lock

class SQLiteCache:
    """
    Simple SQLite Cache implementation
    """
    def __init__(self, db_path="llm_cache.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._init_db()

    def _init_db(self):
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir)
            except:
                pass
            
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
                print(f"[Cache] Failed to init cache db: {e}")

    def get(self, key):
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM llm_cache WHERE key = ?', (key,))
                row = cursor.fetchone()
                conn.close()
                return row[0] if row else None
            except Exception as e:
                print(f"[Cache] Get error: {e}")
                return None

    def set(self, key, value):
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('REPLACE INTO llm_cache (key, value, timestamp) VALUES (?, ?, ?)', 
                             (key, value, time.time()))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[Cache] Set error: {e}")

class LLMClient:
    def __init__(self, config_path="json/llm_config.json"):
        self.config = self._load_config(config_path)
        self.api_key = self.config.get("LLM_APP_KEY")
        self.app_id = self.config.get("LLM_APP_ID")
        self.base_url = self.config.get("LLM_BASE_URL")
        self.model = self.config.get("LLM_MODEL")
        
        # Cache initialization
        self.cache = SQLiteCache("json/llm_cache.db")
        
        # Construct API Key as {APP_ID}.{APP_KEY} if both are present
        if self.app_id and self.api_key:
            auth_token = f"{self.app_id}.{self.api_key}"
        else:
            auth_token = self.api_key

        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

    def _load_config(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[LLMClient] Error loading config: {e}")
            return {}

    def _generate_cache_key(self, model, messages, temperature, max_tokens):
        """Generate a unique cache key for the request"""
        content = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }, sort_keys=True)
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def chat_completion(self, messages, temperature=0.1, max_tokens=2000, json_mode=False):
        # 1. Check Cache
        cache_key = self._generate_cache_key(self.model, messages, temperature, max_tokens)
        cached_res = self.cache.get(cache_key)
        if cached_res:
            return cached_res

        url = f"{self.base_url}/chat/completions"
        
        # Adjust parameters for generic use
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Some APIs support json_object response format
        if json_mode:
            data["response_format"] = {"type": "json_object"}
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=data, timeout=90)
                
                if response.status_code == 200:
                    res_json = response.json()
                    content = res_json['choices'][0]['message']['content']
                    
                    # 2. Save to Cache
                    self.cache.set(cache_key, content)
                    
                    return content
                elif response.status_code == 429:
                    print(f"[LLMClient] Rate Limit Hit (429). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential backoff
                    continue
                else:
                    print(f"[LLMClient] API Error: {response.status_code} - {response.text}")
                    return None
                    
            except Exception as e:
                print(f"[LLMClient] Request Exception: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
        
        return None

if __name__ == "__main__":
    client = LLMClient()
    print("Testing connection...")
    res = client.chat_completion([{"role": "user", "content": "Hello"}])
    print(f"Result: {res}")
