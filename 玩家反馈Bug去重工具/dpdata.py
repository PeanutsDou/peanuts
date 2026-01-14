# -*- coding: utf-8 -*-
import os
import pandas as pd
import glob
import difflib
import json
import time
import concurrent.futures
import math
import re
import copy
try:
    from Emb_data import run_embedding
    from DBS_data import run_clustering
    HAS_NEW_MODULES = True
except ImportError as e:
    HAS_NEW_MODULES = False
    print(f"Warning: Emb_data or DBS_data not found ({e}). Skipping advanced steps.")

# 说明：当前版本取消所有大模型调用，仅保留简单去重与初步语义去重

# Python 2/3 compatibility
try:
    basestring
except NameError:
    basestring = str
    unicode = str

def _script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    """加载配置文件：读取流程相关参数"""
    cfg_path = os.path.join(_script_dir(), "json", "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def find_latest_file(directory):
    """查找目录下最新的xlsx文件"""
    search_pattern = os.path.join(directory, "*.xlsx")
    files = glob.glob(search_pattern)
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def is_similar(s1, s2, threshold=0.75):
    """判断两个字符串是否相似：用于语义去重的基础相似度计算"""
    if not isinstance(s1, basestring): s1 = unicode(s1)
    if not isinstance(s2, basestring): s2 = unicode(s2)
    return difflib.SequenceMatcher(None, s1, s2).ratio() > threshold

def semantic_deduplicate(df, threshold=0.75, is_test=False):
    """对 DataFrame 进行语义去重（基于 difflib，相似度阈值可配置）"""
    if df.empty:
        return df, 0, []
        
    subjects = df["subject"].tolist()
    indices = df.index.tolist()
    
    keep_indices = []
    kept_subjects = []
    removed_logs = []
    
    # 暂时不使用 cluster_id，因为这是预处理步骤
    
    print("正在进行初步语义去重 (阈值: {}), 共 {} 条数据...".format(threshold, len(subjects)))
    
    for i, subj in enumerate(subjects):
        is_dup = False
        for kept_subj in kept_subjects:
            if is_similar(subj, kept_subj, threshold):
                is_dup = True
                if is_test:
                    removed_logs.append({
                        "stage": "semantic_dedup",
                        "removed": subj,
                        "kept": kept_subj,
                        "similarity": "Difflib > {}".format(threshold)
                    })
                break
        
        if not is_dup:
            keep_indices.append(indices[i])
            kept_subjects.append(subj)
            
    return df.loc[keep_indices], len(df) - len(keep_indices), removed_logs

def read_excel_fallback(path):
    """读取 Excel 的回退方法：直接使用 openpyxl 读取，解决引擎问题"""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        data = []
        headers = []
        for row in ws.iter_rows(min_row=1, max_row=1):
            for cell in row:
                headers.append(cell.value)
        try:
            for row in ws.iter_rows(min_row=2, values_only=True):
                data.append(row)
        except TypeError:
            for row in ws.iter_rows(min_row=2):
                data.append([cell.value for cell in row])
        df = pd.DataFrame(data, columns=headers)
        return df
    except Exception as e:
        print("Openpyxl fallback failed: {}".format(e))
        raise e

def run_deduplication(input_data=None):
    """主入口：执行简单去重与初步语义去重，并保存输出"""
    script_dir = _script_dir()
    
    # input_data can be a path (str) or a DataFrame
    df = None
    target_file = None
    
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
        cols_to_drop = ["time", "source_type"]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        print("\n[去重阶段] 接收到内存中的数据 DataFrame ({} 条记录)".format(len(df)))
    else:
        target_file = input_data
        if not target_file:
            cleandata_dir = os.path.join(script_dir, "cleandata")
            if not os.path.isdir(cleandata_dir):
                print("错误: 未找到 cleandata 文件夹。")
                return {"success": False, "message": "No cleandata dir"}
            target_file = find_latest_file(cleandata_dir)
            if not target_file:
                print("错误: cleandata 文件夹中没有数据文件。")
                return {"success": False, "message": "No data file"}
        print("\n[去重阶段] 正在处理文件: {}".format(target_file))
    
    cfg = load_config()
    is_test = cfg.get("isTest", False)
    thresholds = cfg.get("deduplication_thresholds", {"simple_dedup": "first", "semantic_dedup": 0.75})

    try:
        if df is None:
            if isinstance(target_file, str):
                try:
                    target_file = target_file.decode('utf-8')
                except:
                    pass
            try:
                df = pd.read_excel(target_file, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(target_file)
                except:
                    df = read_excel_fallback(target_file)
        
        if "subject" not in df.columns:
            print("错误: 数据表格中没有 'subject' 表头。")
            return {"success": False, "message": "Missing subject column"}

        original_count = len(df)
        try:
            user_input = input("请输入要进入流程的条目数 (直接回车处理全部, 输入数字则随机抽取): ").strip()
            if user_input:
                sample_size = int(user_input)
                if sample_size < original_count:
                    df = df.sample(n=sample_size)
                    print("已随机抽取 {} 条数据进入流程。".format(sample_size))
                else:
                    print("输入数量大于或等于总数，将处理所有 {} 条数据。".format(original_count))
        except Exception:
            print("输入无效，将处理所有数据。")
        original_count = len(df)
        
        # 1. 简单去重 (Exact)
        df_simple_dedup = df.drop_duplicates(subset=["subject"], keep="first")
        simple_removed_count = original_count - len(df_simple_dedup)
        print("简单去重: 发现 {} 条完全重复数据。".format(simple_removed_count))

        # 2. 语义去重 (Similarity)
        semantic_threshold = thresholds.get("semantic_dedup", 0.75)
        df_semantic, semantic_removed_count, semantic_logs = semantic_deduplicate(df_simple_dedup, threshold=semantic_threshold, is_test=is_test)
        print("初步语义去重: 发现 {} 条语义相似数据。".format(semantic_removed_count))

        # 3. 向量化与聚类（Embedding & Clustering）
        cluster_logs = []
        if HAS_NEW_MODULES:
            # 3.1 向量化（保存为 .npy，仅在表内记录索引）
            print("\n[高级处理] 开始文本向量化...")
            df_emb, emb_path = run_embedding(df_semantic, column='subject')
            
            # 3.2 聚类（从 .npy 加载向量矩阵）
            if emb_path:
                print("[高级处理] 开始 DBSCAN 聚类...")
                df_clustered, cluster_logs = run_clustering(df_emb, emb_path=emb_path)
                df_final = df_clustered
            else:
                print("向量化未生成向量文件，跳过聚类。")
                df_final = df_semantic
        else:
            df_final = df_semantic

        final_count = len(df_final)
        print(f"最终结果: {final_count} 条")

        # 4. 保存结果
        timestamp = time.strftime("%Y%m%d")
        cleandata_dir = os.path.join(script_dir, "cleandata")
        if not os.path.isdir(cleandata_dir):
            os.makedirs(cleandata_dir, exist_ok=True)
            
        existing_files = [f for f in os.listdir(cleandata_dir) if "deduplicated_data_{}".format(timestamp) in f]
        version = len(existing_files) + 1
        output_filename = "deduplicated_data_{}_v{}.xlsx".format(timestamp, version)
        output_path = os.path.join(cleandata_dir, output_filename)
        
        # Clean Final Data
        # 清理不需要的列：embedding/embedding_index 等不进入最终 Excel
        cols_to_drop = ['Cluster', 'Core Topic', 'embedding', 'embedding_index']
        df_final_clean = df_final.drop(columns=cols_to_drop, errors='ignore')
        
        if is_test:
            try:
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # 最终去重结果
                    df_final_clean.to_excel(writer, sheet_name='Final_Deduplicated', index=False)
                    # 日志（仅保留语义去重日志）
                    if semantic_logs:
                        df_logs = pd.DataFrame(semantic_logs)
                        df_logs.to_excel(writer, sheet_name='Removed_Logs', index=False)
                    # 聚类日志
                    if cluster_logs:
                        df_clusters = pd.DataFrame(cluster_logs)
                        df_clusters.to_excel(writer, sheet_name='Cluster_Logs', index=False)
                    # 噪点日志 (单独输出 cluster_id == -1 的数据)
                    if 'cluster_id' in df_final_clean.columns:
                        df_noise = df_final_clean[df_final_clean['cluster_id'] == -1]
                        if not df_noise.empty:
                             df_noise.to_excel(writer, sheet_name='Noise_Logs', index=False)
            except Exception as e:
                print(f"Error saving Excel: {e}")
                df_final_clean.to_excel(output_path, index=False)
        else:
            # 非测试模式仅输出最终表
            df_final_clean.to_excel(output_path, index=False)
            
        print("去重后的数据已保存至: {}".format(output_path))
        
        return {
            "success": True,
            "original_count": original_count,
            "dedup_count": final_count,
            "output_path": output_path
        }

    except Exception as e:
        msg = "处理过程中发生错误: {}".format(e)
        print(msg)
        return {"success": False, "message": msg}

if __name__ == "__main__":
    run_deduplication()
