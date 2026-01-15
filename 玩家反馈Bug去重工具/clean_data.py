# -*- coding: utf-8 -*-
import os
import json
import shutil
import pandas as pd
import hashlib
import time
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
        # 如果没有配置文件，这里可以创建一个默认的，或者直接报错
        # 为保证鲁棒性，这里返回空字典或默认值，但在实际流程中应该已有config.json
        print("警告: 配置文件 config.json 不存在。")
        return {}
        
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    """保存配置文件"""
    cfg_dir = os.path.join(_script_dir(), "json")
    cfg_path = os.path.join(cfg_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def compute_config_hash(cfg):
    """计算配置的关键部分哈希值 (排除动态变化的字段)，用于检测配置变更"""
    keys_to_hash = ["rawdata_dir_name", "cleandata_dir_name", "output_filename", "column_mapping"]
    subset = {k: cfg.get(k) for k in keys_to_hash}
    s = json.dumps(subset, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def classify_file(filename):
    """根据文件扩展名初步筛选文件"""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        return None
    return "PendingDetect"

def scan_rawdata(rawdir):
    """扫描原始数据文件夹，返回有效文件列表"""
    items = []
    if not os.path.isdir(rawdir):
        return items
    for name in os.listdir(rawdir):
        path = os.path.join(rawdir, name)
        if not os.path.isfile(path):
            continue
        if name.startswith("~$"): # 忽略Excel临时文件
            continue
        t = classify_file(name)
        if t:
            items.append({"path": path, "name": name, "type": t})
    return items

def prompt_user_for_path():
    """交互式提示用户输入文件路径（当自动扫描失败时）"""
    print("未在原始数据文件夹找到可用数据，请输入数据文件的绝对路径（可输入多个，使用分号;分隔）：")
    raw = input().strip()
    if not raw:
        print("未输入任何内容")
        return None
    parts = [x.strip().strip('"') for x in raw.split(";") if x.strip().strip('"')]
    items = []
    for p in parts:
        if not os.path.isabs(p):
            print(f"输入路径不是绝对路径: {p}")
            continue
        if os.path.isdir(p):
            dir_items = scan_rawdata(p)
            items.extend(dir_items)
        elif os.path.isfile(p):
            name = os.path.basename(p)
            t = classify_file(name)
            if t:
                items.append({"path": p, "name": name, "type": t})
    return items if items else None

def read_file_content(path):
    """读取文件内容并返回DataFrame"""
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        return df
    except Exception as e:
        print(f"读取文件失败: {path}, 错误: {e}")
        return None

def detect_type_and_extract(df, mapping, filename):
    """
    根据列名自动探测数据类型并提取数据
    返回: (type_name, extracted_df)
    """
    columns = set(df.columns)
    
    for type_name, config in mapping.items():
        if type_name.startswith("_"): continue # 跳过注释字段

        found_cols = {}
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
            # 提取并重命名
            selected_cols = list(found_cols.values())
            rename_map = {v: k for k, v in found_cols.items()}
            
            extracted_df = df[selected_cols].copy()
            extracted_df.rename(columns=rename_map, inplace=True)
            return type_name, extracted_df

    return None, None

def process_data(items, cfg):
    """
    核心处理逻辑：读取 -> 识别 -> 增量过滤 -> 合并
    """
    start_time = time.time()
    all_data = []
    mapping = cfg.get("column_mapping", {})
    records = cfg.get("last_processed_records", {})
    new_records = records.copy()
    
    # 尝试引入 tqdm 显示进度
    try:
        from tqdm import tqdm
        iter_items = tqdm(items, desc="正在清洗文件")
    except ImportError:
        iter_items = items
        print("正在处理文件...")

    for item in iter_items:
        fpath = item["path"]
        fname = item["name"]
        
        df = read_file_content(fpath)
        if df is None or df.empty:
            continue

        # 自动探测类型并提取
        ftype, extract_df = detect_type_and_extract(df, mapping, fname)
        
        if not ftype:
            # 仅在非tqdm模式或详细日志模式下打印，避免刷屏
            # print(f"文件 {fname} 未能匹配列配置，跳过。")
            continue

        item["type"] = ftype
        
        # 增量过滤逻辑
        last_time_str = records.get(ftype, "2000-01-01 00:00:00")
        try:
            last_time = pd.to_datetime(last_time_str)
        except:
            last_time = pd.to_datetime("2000-01-01 00:00:00")

        # 转换时间列
        try:
            extract_df["time"] = pd.to_datetime(extract_df["time"])
        except Exception:
            continue

        # 筛选新数据
        is_test = cfg.get("isTest", False)
        if is_test:
            new_df = extract_df.copy()
        else:
            new_df = extract_df[extract_df["time"] > last_time].copy()
        
        if new_df.empty:
            continue
            
        # 更新记录
        # 注意: 只有在非测试模式下才更新时间戳，否则会造成数据丢失
        if not is_test:
            current_max_time = new_df["time"].max()
            current_record_time = pd.to_datetime(new_records.get(ftype, "2000-01-01 00:00:00"))
            if current_max_time > current_record_time:
                new_records[ftype] = str(current_max_time)

        new_df["source_type"] = ftype
        all_data.append(new_df)

    if not all_data:
        return pd.DataFrame(), new_records
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 新增: 任务ID (task_id)
    # 格式: 全局自增ID，确保不乱序
    # 为了方便后续吸附，ID从10001开始
    task_ids = range(10001, 10001 + len(final_df))
    final_df.insert(0, 'task_id', task_ids)
    
    end_time = time.time()
    print(f"清洗阶段耗时: {end_time - start_time:.2f}秒")
    
    return final_df, new_records

def run():
    """clean_data 模块主入口"""
    cfg = load_config()
    
    # 检查配置变更
    current_hash = compute_config_hash(cfg)
    stored_hash = cfg.get("config_hash")
    
    if stored_hash != current_hash:
        print("\n[Config] 配置变更，重置增量记录，将全量扫描。")
        cfg["last_processed_records"] = {
            "QData": "2000-01-01 00:00:00",
            "BugFeedback": "2000-01-01 00:00:00"
        }
        cfg["config_hash"] = current_hash
        save_config(cfg)
    
    rawdir = os.path.join(_script_dir(), cfg.get("rawdata_dir_name", "rawdata"))
    
    # 1. 获取文件列表
    items = scan_rawdata(rawdir)
    source = "folder"
    
    if not items:
        user_items = prompt_user_for_path()
        if user_items:
            items = user_items
            source = "user_input"
            # 复制到 rawdata 备份
            if not os.path.isdir(rawdir):
                os.makedirs(rawdir, exist_ok=True)
            for it in user_items:
                src = it["path"]
                dst = os.path.join(rawdir, it["name"])
                if os.path.abspath(src) != os.path.abspath(dst):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

    if not items:
        return {"success": False, "message": "未找到数据文件", "rawdata_dir": rawdir}

    # 2. 清洗与整合
    cleaned_df, new_records = process_data(items, cfg)
    
    if cleaned_df.empty:
         return {
            "success": True,
            "message": "未发现新数据",
            "data": items,
            "rawdata_dir": rawdir,
            "config_path": os.path.join(_script_dir(), "json", "config.json")
        }

    # 3. 准备输出
    is_test = cfg.get("isTest", False)
    if not is_test:
        cfg["last_processed_records"] = new_records
        save_config(cfg)

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
