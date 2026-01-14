# -*- coding: utf-8 -*-
import os
import json
import shutil
import pandas as pd
import hashlib
from datetime import datetime

# 允许的文件扩展名
ALLOWED_EXTS = {"xlsx", "csv"}

def _script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    """加载配置文件，若不存在则创建默认配置"""
    cfg_dir = os.path.join(_script_dir(), "json")
    cfg_path = os.path.join(cfg_dir, "config.json")
    if not os.path.isdir(cfg_dir):
        os.makedirs(cfg_dir, exist_ok=True)
    if not os.path.isfile(cfg_path):
        # 默认配置
        default_cfg = {
            "rawdata_dir_name": "rawdata",
            "cleandata_dir_name": "cleandata",
            "output_filename": "cleaned_bug_feedback.xlsx",
            "isTest": True,
            "deduplication_thresholds": {
                "simple_dedup": "first",
                "semantic_dedup": 0.75,
                "llm_dedup": 0.90
            },
            "column_mapping": {
                "QData": {
                    "subject": ["内容"],
                    "image": ["图片"],
                    "time": "时间"
                },
                "BugFeedback": {
                    "subject": ["主题"],
                    "image": ["URL"],
                    "time": "反馈时间"
                }
            },
            "last_processed_records": {
                "QData": "2000-01-01 00:00:00",
                "BugFeedback": "2000-01-01 00:00:00"
            }
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(default_cfg, f, ensure_ascii=False, indent=2)
        return default_cfg
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    """保存配置文件"""
    cfg_dir = os.path.join(_script_dir(), "json")
    cfg_path = os.path.join(cfg_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def compute_config_hash(cfg):
    """计算配置的关键部分哈希值 (排除动态变化的字段)"""
    # 提取关键字段
    keys_to_hash = ["rawdata_dir_name", "cleandata_dir_name", "output_filename", "column_mapping"]
    subset = {k: cfg.get(k) for k in keys_to_hash}
    
    # 序列化并计算哈希
    s = json.dumps(subset, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def classify_file(filename):
    """根据文件名初步筛选文件"""
    lower = filename.lower()
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        return None
    # 不再限制文件名关键词，而是让后续的 detect_type_and_extract 根据列名判断
    return "PendingDetect"

def scan_rawdata(rawdir):
    """扫描原始数据文件夹"""
    items = []
    if not os.path.isdir(rawdir):
        return items
    for name in os.listdir(rawdir):
        path = os.path.join(rawdir, name)
        if not os.path.isfile(path):
            continue
        t = classify_file(name)
        if t:
            items.append({"path": path, "name": name, "type": t})
    return items

def prompt_user_for_path():
    """提示用户输入文件或文件夹路径"""
    print("未在原始数据文件夹找到可用数据，请输入数据文件的绝对路径（可输入多个，使用分号;分隔，或输入包含所有数据文件的文件夹绝对路径）：")
    raw = input().strip()
    if not raw:
        print("未输入任何内容")
        return None
    parts = [x.strip().strip('"') for x in raw.split(";") if x.strip().strip('"')]
    items = []
    for p in parts:
        if not os.path.isabs(p):
            print("输入路径不是绝对路径: {}".format(p))
            continue
        if os.path.isdir(p):
            dir_items = scan_rawdata(p)
            if not dir_items:
                print("文件夹中未找到可用数据文件: {}".format(p))
            else:
                items.extend(dir_items)
        elif os.path.isfile(p):
            name = os.path.basename(p)
            t = classify_file(name)
            if not t:
                print("输入文件不符合数据类型要求: {}".format(p))
            else:
                items.append({"path": p, "name": name, "type": t})
        else:
            print("输入路径不存在: {}".format(p))
    if not items:
        print("未从输入路径中获取到任何有效数据文件")
        return None
    return items

def read_file_content(path):
    """读取文件内容并返回DataFrame"""
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        return df
    except Exception as e:
        print("读取文件失败: {}, 错误: {}".format(path, e))
        return None

def detect_type_and_extract(df, mapping, filename):
    """
    根据列名自动探测数据类型并提取数据
    返回: (type_name, extracted_df, time_col_name) 或 (None, None, None)
    """
    columns = set(df.columns)
    
    for type_name, config in mapping.items():
        # 1. 检查所有配置的列是否都存在（至少匹配到一个候选列）
        found_cols = {} # {config_key: actual_col_name}
        all_matched = True
        
        for key, candidates in config.items():
            if key == "time":
                # 时间列是单独的字符串
                if candidates in columns:
                    found_cols[key] = candidates
                else:
                    all_matched = False
                    break
            else:
                # 其他列是候选列表
                match = None
                for cand in candidates:
                    if cand in columns:
                        match = cand
                        break
                if match:
                    found_cols[key] = match
                else:
                    all_matched = False
                    break
        
        if all_matched:
            print("文件 {} 识别为类型: {}".format(filename, type_name))
            
            # 提取并重命名
            # df selection: list of actual column names
            # rename dict: {actual_col_name: config_key}
            
            selected_cols = list(found_cols.values())
            rename_map = {v: k for k, v in found_cols.items()}
            
            extracted_df = df[selected_cols].copy()
            extracted_df.rename(columns=rename_map, inplace=True)
            
            time_col_key = "time" # After rename, the time column is named "time"
            
            return type_name, extracted_df, time_col_key

    return None, None, None

def process_data(items, cfg):
    """处理数据：提取、重命名、去重"""
    all_data = []
    mapping = cfg.get("column_mapping", {})
    records = cfg.get("last_processed_records", {})
    new_records = records.copy()

    for item in items:
        fpath = item["path"]
        fname = item["name"]
        
        df = read_file_content(fpath)
        if df is None or df.empty:
            continue

        # 自动探测类型并提取
        ftype, extract_df, time_col_raw = detect_type_and_extract(df, mapping, fname)
        
        if not ftype:
            print("文件 {} 未能匹配任何已知的列配置，跳过。".format(fname))
            print("  - 现有列名: {}".format(list(df.columns)))
            continue

        # 更新 item 的类型以便后续显示
        item["type"] = ftype
            
        last_time_str = records.get(ftype, "2000-01-01 00:00:00")
        try:
            last_time = pd.to_datetime(last_time_str)
        except:
            last_time = pd.to_datetime("2000-01-01 00:00:00")

        # 统一列名 (已在 detect_type_and_extract 中完成重命名，这里不需要再赋值 columns)
        # extract_df.columns = ["subject", "image", "time"] 
        
        # 转换时间列
        try:
            extract_df["time"] = pd.to_datetime(extract_df["time"])
        except Exception as e:
            print("文件 {} 时间列转换失败: {}".format(fname, e))
            continue

        # 筛选新数据
        is_test = cfg.get("isTest", False)
        
        if is_test:
            # isTest=True 时，不按时间过滤，处理所有数据
            new_df = extract_df.copy()
            print("测试模式 (isTest=True): 跳过时间过滤，处理所有数据。")
        else:
            new_df = extract_df[extract_df["time"] > last_time].copy()
        
        if new_df.empty:
            print("文件 {} ({}) 没有新数据 (最新时间: {})".format(fname, ftype, extract_df['time'].max()))
            continue
            
        print("从文件 {} ({}) 提取到 {} 条新数据".format(fname, ftype, len(new_df)))

        # 更新该类型的最新时间记录
        current_max_time = new_df["time"].max()
        current_record_time = pd.to_datetime(new_records.get(ftype, "2000-01-01 00:00:00"))
        if current_max_time > current_record_time:
            new_records[ftype] = str(current_max_time)

        new_df["source_type"] = ftype # 标记来源
        all_data.append(new_df)

    if not all_data:
        return pd.DataFrame(), new_records
    
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df, new_records

def run():
    cfg = load_config()
    
    # 检查配置是否变更
    current_hash = compute_config_hash(cfg)
    stored_hash = cfg.get("config_hash")
    
    if stored_hash != current_hash:
        print("\n[Config Change] 检测到关键配置已变更，重置增量更新记录，将重新扫描所有数据。")
        # 重置时间记录
        cfg["last_processed_records"] = {
            "QData": "2000-01-01 00:00:00",
            "BugFeedback": "2000-01-01 00:00:00"
        }
        # 更新哈希
        cfg["config_hash"] = current_hash
        save_config(cfg)
    
    rawdir = os.path.join(_script_dir(), cfg.get("rawdata_dir_name", "rawdata"))
    
    # 1. 获取文件列表
    items = scan_rawdata(rawdir)
    success = False
    source = "folder"
    
    if items:
        success = True
    else:
        user_items = prompt_user_for_path()
        if user_items:
            success = True
            items = user_items
            source = "user_input"
            # 复制到 rawdata
            if not os.path.isdir(rawdir):
                os.makedirs(rawdir, exist_ok=True)
            for it in user_items:
                src = it["path"]
                dst = os.path.join(rawdir, it["name"])
                if os.path.abspath(src) == os.path.abspath(dst):
                    continue
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print("复制到rawdata失败: {} -> {}: {}".format(src, dst, e))

    if not success:
        return {
            "success": False, 
            "message": "未找到数据文件",
            "rawdata_dir": rawdir,
            "config_path": os.path.join(_script_dir(), "json", "config.json")
        }

    # 2. 清洗与整合数据
    cleaned_df, new_records = process_data(items, cfg)
    
    if cleaned_df.empty:
         return {
            "success": True,
            "message": "未发现新数据需要更新",
            "data": items,
            "rawdata_dir": rawdir,
            "config_path": os.path.join(_script_dir(), "json", "config.json")
        }

    # 3. 保存结果
    clean_dir = os.path.join(_script_dir(), cfg.get("cleandata_dir_name", "cleandata"))
    if not os.path.isdir(clean_dir):
        os.makedirs(clean_dir, exist_ok=True)
        
    is_test = cfg.get("isTest", False)
    
    # 4. 更新配置中的时间记录 (仅在非测试模式下更新)
    if not is_test:
        cfg["last_processed_records"] = new_records
        save_config(cfg)
    else:
        print("测试模式 (isTest=True): 不更新时间戳记录。")

    return {
        "success": True,
        "source": source,
        "data": items, 
        "rawdata_dir": rawdir,
        "config_path": os.path.join(_script_dir(), "json", "config.json"),
        "cleaned_df": cleaned_df,
        "new_count": len(cleaned_df),
        "is_test": is_test
    }

if __name__ == "__main__":
    run()
