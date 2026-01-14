# -*- coding: utf-8 -*-
# @Author: Xuguoliang
# @Date:   2025-12-17 10:00:00
# @Version: 1.2.0

import os
import json
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import asdict
from typing import List

from config import Config
from preprocessor import Preprocessor
from classifier import Classifier
from ticket_manager import TicketManager, Ticket
from llm_client import LLMClient, LLMConfig

def process_single_feedback(classifier, ticket_manager, row, content_col):
    """
    单个反馈的处理逻辑 (用于多线程)
    """
    content = row['cleaned_content']
    original = row[content_col]
    
    # AI 分类
    ai_result = classifier.classify(content)
    
    # 生成工单
    ticket = ticket_manager.create_ticket(content, original, ai_result)
    return ticket

def process_clustering_batch(classifier, tickets: List[Ticket], options: List[str]) -> List[Ticket]:
    """
    批量处理子子分类归属
    """
    feedbacks = [t.cleaned_feedback for t in tickets]
    results = classifier.batch_assign_sub_subcategories(feedbacks, options)
    
    for t, label in zip(tickets, results):
        t.sub_subcategory = label
        
    return tickets

def main():
    parser = argparse.ArgumentParser(description="玩家反馈数据清洗与分析工作流")
    parser.add_argument("input_file", nargs="?", help="待分析的Excel文件路径")
    args = parser.parse_args()

    print(">>> 启动玩家反馈数据清洗与分析工作流")
    cfg = Config()
    
    # 允许命令行参数覆盖配置文件
    if args.input_file:
        cfg.DATA_FILE = args.input_file
    
    # 1. 加载数据
    if not cfg.DATA_FILE:
        # 如果未通过命令行参数指定，则请求用户输入
        user_input = input("请输入待分析的数据文件路径 (.xlsx 或 .csv): ").strip()
        # 移除可能的引号 (当用户直接拖拽文件到终端时可能会带引号)
        if user_input.startswith('"') and user_input.endswith('"'):
            user_input = user_input[1:-1]
        elif user_input.startswith("'") and user_input.endswith("'"):
            user_input = user_input[1:-1]
        cfg.DATA_FILE = user_input

    if not cfg.DATA_FILE or not os.path.exists(cfg.DATA_FILE):
        print(f"Error: 文件不存在或未指定 {cfg.DATA_FILE}")
        return
        
    try:
        # 根据后缀名判断读取方式
        _, ext = os.path.splitext(cfg.DATA_FILE)
        ext = ext.lower()
        
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(cfg.DATA_FILE)
        elif ext == '.csv':
            # 尝试检测编码
            try:
                df = pd.read_csv(cfg.DATA_FILE, encoding='utf-8')
            except UnicodeDecodeError:
                print("Warning: UTF-8解码失败，尝试GBK...")
                df = pd.read_csv(cfg.DATA_FILE, encoding='gbk')
        else:
            print(f"Error: 不支持的文件格式 {ext}，请提供 .xlsx 或 .csv 文件")
            return

        print(f"成功读取原始数据: {len(df)} 条")
        
        # 检查列名
        content_col = cfg.COL_CONTENT
        if content_col not in df.columns:
            print(f"Warning: 未找到列 '{content_col}'，尝试使用第1列")
            content_col = df.columns[0]
            
    except Exception as e:
        print(f"Error: 读取文件失败 - {e}")
        return

    # 2. 数据预处理
    preprocessor = Preprocessor(cfg)
    
    # 清洗
    print(">>> Phase 1: 数据清洗...")
    df['cleaned_content'] = df[content_col].apply(preprocessor.clean_text)
    
    # 去重
    print(">>> Phase 1: 数据去重...")
    df_unique = preprocessor.remove_duplicates(df, 'cleaned_content')
    print(f"去重后剩余: {len(df_unique)} 条 (过滤掉 {len(df) - len(df_unique)} 条)")

    # 3. 智能分类 & 工单生成
    print(">>> Phase 2 & 3: 智能分类与工单生成...")
    
    # Checkpoint logic for Phase 2&3
    CHECKPOINT_FILE = "checkpoint_classification.jsonl"
    processed_tickets = {} # map cleaned_content -> Ticket
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading classification checkpoint from {CHECKPOINT_FILE}...")
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    t = Ticket(**data)
                    processed_tickets[t.cleaned_feedback] = t
            print(f"Loaded {len(processed_tickets)} tickets from checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Identify remaining items
    remaining_indices = []
    for idx, row in df_unique.iterrows():
        if row['cleaned_content'] not in processed_tickets:
            remaining_indices.append(idx)
            
    process_df = df_unique.loc[remaining_indices]
    print(f"需要处理: {len(process_df)} 条 (已处理: {len(processed_tickets)})")

    llm_config = LLMConfig(
        app_id=cfg.LLM_APP_ID,
        app_key=cfg.LLM_APP_KEY,
        base_url=cfg.LLM_BASE_URL,
        model=cfg.LLM_MODEL,
        cache_file=cfg.LLM_CACHE_FILE
    )
    llm_client = LLMClient(llm_config)
    classifier = Classifier(llm_client, cfg)
    ticket_manager = TicketManager()

    if not process_df.empty:
        print(f"正在处理剩余 {len(process_df)} 条数据...")
        
        # Open checkpoint file in append mode
        with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as cp_file:
            with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
                futures = []
                for index, row in process_df.iterrows():
                    futures.append(executor.submit(
                        process_single_feedback, 
                        classifier, 
                        ticket_manager, 
                        row, 
                        content_col
                    ))
                    
                for future in tqdm(as_completed(futures), total=len(futures), desc="AI Analysis"):
                    try:
                        ticket = future.result()
                        # Save to memory
                        processed_tickets[ticket.cleaned_feedback] = ticket
                        # Save to file
                        cp_file.write(json.dumps(asdict(ticket), ensure_ascii=False) + "\n")
                        cp_file.flush()
                    except Exception as e:
                        print(f"Error processing item: {e}")

    tickets = list(processed_tickets.values())

    # 4. 细粒度聚类 (Sub-subcategory Clustering)
    print(">>> Phase 3.5: 细粒度聚类 (Sub-subcategory Clustering)...")
    
    # Checkpoint logic for Phase 3.5
    CLUSTERING_CHECKPOINT_FILE = "checkpoint_clustering.jsonl"
    clustered_tickets_map = {} # cleaned_content -> Ticket (with sub_subcategory)

    if os.path.exists(CLUSTERING_CHECKPOINT_FILE):
        print(f"Loading clustering checkpoint from {CLUSTERING_CHECKPOINT_FILE}...")
        try:
            with open(CLUSTERING_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    t = Ticket(**data)
                    clustered_tickets_map[t.cleaned_feedback] = t
            print(f"Loaded {len(clustered_tickets_map)} clustered tickets.")
        except Exception as e:
            print(f"Error loading clustering checkpoint: {e}")

    # 按 category -> subcategory 分组
    grouped_tickets = {}
    for t in tickets:
        key = (t.category, t.subcategory)
        if key not in grouped_tickets:
            grouped_tickets[key] = []
        grouped_tickets[key].append(t)
    
    final_tickets = []
    
    # 进度条
    pbar = tqdm(total=len(grouped_tickets), desc="Clustering Groups")
    
    # Open clustering checkpoint in append mode
    with open(CLUSTERING_CHECKPOINT_FILE, 'a', encoding='utf-8') as cp_file:
        for (cat, sub), group in grouped_tickets.items():
            # Check if this group is already fully clustered
            all_clustered = True
            group_results = []
            tickets_to_process = []
            
            for t in group:
                if t.cleaned_feedback in clustered_tickets_map:
                    group_results.append(clustered_tickets_map[t.cleaned_feedback])
                else:
                    all_clustered = False
                    tickets_to_process.append(t)
            
            # If all are already processed, add them and continue
            if all_clustered:
                final_tickets.extend(group_results)
                pbar.update(1)
                continue
            
            # If some are processed, add them first
            if group_results:
                 final_tickets.extend(group_results)

            # Process remaining tickets in this group
            if not tickets_to_process:
                pbar.update(1)
                continue
            
            # 如果样本太少，跳过聚类，直接归为 "其他"
            if len(tickets_to_process) < 5 and len(group) < 5: # check total group size too
                for t in tickets_to_process:
                    t.sub_subcategory = "其他"
                    final_tickets.append(t)
                    # Save checkpoint
                    cp_file.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
                cp_file.flush()
                pbar.update(1)
                continue
                
            # 1. 提炼子子分类 (Use all feedbacks in group to generate options, not just unprocessed ones)
            # Actually, we should probably re-use options if we had them? 
            # But simpler to just regenerate or use existing logic. 
            # The prompt logic uses a sample.
            all_feedbacks_in_group = [t.cleaned_feedback for t in group]
            sub_sub_options = classifier.generate_sub_subcategories(cat, sub, all_feedbacks_in_group)
            
            if not sub_sub_options:
                # 提炼失败，全部归为其他
                for t in tickets_to_process:
                    t.sub_subcategory = "其他"
                    final_tickets.append(t)
                    # Save checkpoint
                    cp_file.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
                cp_file.flush()
                pbar.update(1)
                continue
                
            # 2. 批量为每条反馈分配子子分类
            # Split tickets_to_process into batches
            batches = [tickets_to_process[i:i + cfg.BATCH_SIZE] for i in range(0, len(tickets_to_process), cfg.BATCH_SIZE)]
            
            with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as sub_executor:
                sub_futures = [sub_executor.submit(process_clustering_batch, classifier, batch, sub_sub_options) for batch in batches]
                
                for f in as_completed(sub_futures):
                    try:
                        batch_results = f.result()
                        for res_ticket in batch_results:
                            final_tickets.append(res_ticket)
                            # Save checkpoint
                            cp_file.write(json.dumps(asdict(res_ticket), ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"Batch processing error: {e}")
            
            cp_file.flush()
            pbar.update(1)
    
    pbar.close()
    
    # 5. 生成交付物
    print(">>> Phase 4 & 5: 生成交付物...")
    ticket_manager.generate_report(final_tickets, cfg.OUTPUT_FILE)
    ticket_manager.generate_daily_stats(final_tickets)
    
    print(f">>> 工作流结束! 报告位置: {cfg.OUTPUT_FILE}")

if __name__ == "__main__":
    main()
