# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import random
import json
import requests
import re
import time

# 添加父目录到系统路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from AI_analyze_data import load_config, call_llm_dedup

# DeepSeek 配置 (硬编码作为基准)
DS_CONFIG = {
    "app_id": "fob3ubx6yv5jxjag",
    "app_key": "x55ll082jpilelrdvv57lt1ocoao726v",
    "base_url": "https://aigw.nie.netease.com/v1",
    "model": "deepseek-v3.2-latest",
    "timeout": 120
}

def call_deepseek_dedup_direct(subjects):
    """
    直接调用 DeepSeek 进行去重，作为基准测试
    """
    print("正在调用 DeepSeek (基准) 进行分析...")
    app_id = DS_CONFIG["app_id"]
    app_key = DS_CONFIG["app_key"]
    base_url = DS_CONFIG["base_url"]
    model = DS_CONFIG["model"]
    
    auth_key = f"{app_id}.{app_key}" if app_id else app_key
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }
    
    content_text = "\n".join([f"{i+1}. {str(s)[:200]}" for i, s in enumerate(subjects)])
    
    # 使用与 AI_analyze_data.py 相同的 Prompt 结构，确保公平性，但模型更强
    prompt = f"""
你是一位资深游戏质量保证(QA)专家。以下是一组玩家反馈的Bug或建议，其中可能包含语义重复的条目。
请仔细分析语义，去除重复内容，只保留不重复的反馈条目。

**去重原则（Critical Principles）：**
1. **区分具体场景**：涉及不同副本（如“万国酒店” vs “迷失乐园”）、不同道具或不同怪物的反馈，即使现象相似（如“卡顿”、“太黑”），也**绝不**视为重复，必须保留。
2. **合并同义表述**：只有当两条反馈描述的是**同一个具体问题**（Same Specific Issue）时才合并。
3. **保留信息量最大的条目**：在判定为重复的一组中，**必须**保留描述最详细的那一条作为代表。
4. **忽略情绪词**：忽略“恶心”、“垃圾”等情绪性词汇。
5. **多问题拆分**：如果一条反馈包含多个独立问题，只要其中任一问题在其他条目中未出现，该条目就应保留。

请直接以 JSON 字符串列表格式返回结果，列表元素为保留下来的原始反馈文本（请保持原文，不要改写）。
例如：["反馈1原文", "反馈2原文"]。

反馈列表：
{content_text}
"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        
        # 清理 Markdown
        clean_content = re.sub(r'^```json\s*', '', content.strip())
        clean_content = re.sub(r'^```\s*', '', clean_content)
        clean_content = re.sub(r'\s*```$', '', clean_content)
        
        deduped = json.loads(clean_content)
        if isinstance(deduped, list):
            return deduped
        else:
            print(f"DeepSeek returned non-list JSON: {content}")
            return subjects
            
    except Exception as e:
        print(f"DeepSeek call failed: {e}")
        return subjects

def run_comparison():
    # 1. 获取用户输入的数据文件路径
    while True:
        data_path = input("请输入测试数据文件的绝对路径: ").strip()
        # 去除可能存在的引号
        data_path = data_path.replace('"', '').replace("'", "")
        
        if not data_path:
            print("路径不能为空，请重新输入。")
            continue
            
        if os.path.exists(data_path) and os.path.isfile(data_path):
            break
        else:
            print(f"文件不存在或无法访问: {data_path}")
            print("请检查路径是否正确。")

    print(f"读取文件: {data_path}")
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return
    
    if 'subject' not in df.columns:
        print("未找到 subject 列")
        return

    all_subjects = df['subject'].dropna().astype(str).tolist()
    total_count = len(all_subjects)
    print(f"数据总条数: {total_count}")
    
    # 2. 获取用户输入的抽样数量
    default_sample = 30
    user_input = input(f"请输入随机抽样数量 (默认 {default_sample}): ").strip()
    
    sample_size = default_sample
    if user_input:
        try:
            val = int(user_input)
            if val > 0:
                sample_size = val
            else:
                print("输入必须为正整数，使用默认值。")
        except ValueError:
            print("输入无效，使用默认值。")
            
    if sample_size > total_count:
        print(f"输入数量 {sample_size} 大于总数，将使用全部数据。")
        sample_size = total_count

    if sample_size > 0:
        sample_subjects = random.sample(all_subjects, sample_size)
    else:
        sample_subjects = []
    
    print(f"\n抽样数据 ({len(sample_subjects)} 条):")
    for i, s in enumerate(sample_subjects):
        print(f"{i+1}. {s}")
        
    # 3. 执行对比测试
    
    # A. 基准组 (DeepSeek)
    start_time = time.time()
    benchmark_result = call_deepseek_dedup_direct(sample_subjects)
    ds_time = time.time() - start_time
    print(f"DeepSeek 处理完成，保留 {len(benchmark_result)} 条，耗时 {ds_time:.2f}s")

    # B. 常规组 (Main Flow / GLM)
    # 加载配置，模拟 main 流程调用 call_llm_dedup
    cfg = load_config()
    print(f"\n正在调用常规流程 (Model: {cfg.get('llm_settings', {}).get('model', 'Unknown')}) 进行分析...")
    
    start_time = time.time()
    normal_result = call_llm_dedup(sample_subjects, cfg)
    normal_time = time.time() - start_time
    print(f"常规流程处理完成，保留 {len(normal_result)} 条，耗时 {normal_time:.2f}s")
    
    # 4. 结果分析
    
    # 转为集合进行对比 (去除首尾空格以防微小差异)
    bench_set = set([s.strip() for s in benchmark_result])
    normal_set = set([s.strip() for s in normal_result])
    original_set = set([s.strip() for s in sample_subjects])
    
    # 正常保留 (两边都保留)
    correct_kept = bench_set.intersection(normal_set)
    
    # 漏删 (DeepSeek 删了，但常规没删 -> 常规保留了不该保留的)
    # 定义：In Normal Set BUT NOT in Benchmark Set
    # 注意：这其实是“常规保留了 Benchmark 认为应该剔除的”
    missed_dedup = normal_set - bench_set
    
    # 误删 (DeepSeek 没删，但常规删了 -> 常规删了不该删的)
    # 定义：In Benchmark Set BUT NOT in Normal Set
    over_dedup = bench_set - normal_set
    
    # 正常剔除 (两边都剔除)
    # Original - (Bench Union Normal) ? No.
    # Correctly Removed: (Original - Bench) AND (Original - Normal)
    # i.e., Items NOT in Bench AND NOT in Normal
    bench_removed = original_set - bench_set
    normal_removed = original_set - normal_set
    correct_removed = bench_removed.intersection(normal_removed)

    # 准确率计算
    # 简单定义：常规流程做出的决策（保留/剔除）与 DeepSeek 一致的比例
    # 一致的决策 = (两边都保留的条目数 + 两边都剔除的条目数) / 总条目数
    match_count = len(correct_kept) + len(correct_removed)
    accuracy = (match_count / len(sample_subjects)) * 100 if sample_subjects else 0
    
    # 5. 输出报告
    report_path = os.path.join(current_dir, "analysis_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== AI 去重对比分析报告 ===\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数量: {len(sample_subjects)}\n")
        f.write(f"基准模型 (Benchmark): {DS_CONFIG['model']}\n")
        f.write(f"常规模型 (Normal): {cfg.get('llm_settings', {}).get('model', 'Unknown')}\n")
        f.write("-" * 50 + "\n")
        
        f.write(f"基准保留: {len(bench_set)} 条\n")
        f.write(f"常规保留: {len(normal_set)} 条\n")
        f.write(f"常规流程准确率: {accuracy:.2f}%\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("【漏查重条目 (Missed Deduplication)】\n")
        f.write("(DeepSeek 认为重复应删，但常规流程保留了)\n")
        if missed_dedup:
            for i, s in enumerate(missed_dedup):
                f.write(f"{i+1}. {s}\n")
        else:
            f.write("无\n")
        f.write("\n")
            
        f.write("【多剔除条目 (Over Deduplication)】\n")
        f.write("(DeepSeek 认为应保留，但常规流程删掉了)\n")
        if over_dedup:
            for i, s in enumerate(over_dedup):
                f.write(f"{i+1}. {s}\n")
        else:
            f.write("无\n")
        f.write("\n")
        
        f.write("【核对正常 - 正确保留 (Correctly Kept)】\n")
        if correct_kept:
            for i, s in enumerate(correct_kept):
                f.write(f"{i+1}. {s}\n")
        else:
            f.write("无\n")
        f.write("\n")
        
        f.write("【核对正常 - 正确剔除 (Correctly Removed)】\n")
        if correct_removed:
            for i, s in enumerate(correct_removed):
                f.write(f"{i+1}. {s}\n")
        else:
            f.write("无\n")
        f.write("\n")
            
        f.write("=" * 50 + "\n")
        f.write("原始抽样列表:\n")
        for i, s in enumerate(sample_subjects):
            f.write(f"{i+1}. {s}\n")

    print(f"\n分析报告已生成: {report_path}")
    print(f"准确率: {accuracy:.2f}%")
    
    # 打印差异概览到控制台
    print("\n=== 差异概览 ===")
    print(f"漏删 (应删未删): {len(missed_dedup)}")
    print(f"误删 (应留未留): {len(over_dedup)}")

if __name__ == "__main__":
    run_comparison()
