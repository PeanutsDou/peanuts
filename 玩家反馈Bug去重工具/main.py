# -*- coding: utf-8 -*-
from clean_data import run as run_cleaning
from dpdata import run_deduplication, semantic_deduplicate
from AI_analyze_data import run_ai_analysis
from translate_data import run_normalization
import sys
import pandas as pd
import time
import os

def main():
    """
    主程序入口
    按顺序执行：数据清洗 -> 规范化(可选) -> 基础去重与聚类 -> AI语义分析
    """
    total_start_time = time.time()
    print("=" * 40)
    print("       玩家反馈Bug去重工具 - 主流程启动       ")
    print("=" * 40)
    
    # 1. 执行数据清洗
    print("\n" + "=" * 30)
    print("步骤 1: 数据清洗 (Data Cleaning)")
    print("=" * 30)
    step1_start = time.time()
    
    cleaning_result = run_cleaning()
    
    cleaned_df = None
    
    if cleaning_result["success"]:
        if "message" in cleaning_result:
            print(f"\n提示: {cleaning_result['message']}")
        else:
            print("\n数据处理成功！")
            if "new_count" in cleaning_result:
                print(f"本次新增清洗数据: {cleaning_result['new_count']} 条")
            if "cleaned_df" in cleaning_result:
                cleaned_df = cleaning_result['cleaned_df']
                print("清洗后数据已加载到内存，准备进行下一步处理...")
                
        # 打印源文件信息
        if "data" in cleaning_result and cleaning_result["data"]:
            print("\n处理的源文件列表:")
            for it in cleaning_result["data"]:
                print(f'- {it["name"]} ({it["type"]})')
    else:
        print(f"\n错误: 清洗失败 - {cleaning_result.get('message', '未知错误')}")
        return
        
    print(f"原始数据目录: {cleaning_result.get('rawdata_dir')}")
    print(f"配置文件路径: {cleaning_result.get('config_path')}")
    
    step1_end = time.time()
    print(f"步骤 1 耗时: {step1_end - step1_start:.2f} 秒")
    
    # 2. 执行数据去重 (包含规范化)
    print("\n" + "=" * 30)
    print("步骤 2: 数据去重 (Deduplication)")
    print("=" * 30)
    step2_start = time.time()
    
    dedup_input = None
    translation_logs = None
    keyword_dedup_logs = None
    image_check_logs = None
    
    if cleaned_df is not None and not cleaned_df.empty:
        # 子步骤：规范化处理
        print("\n>>> 子步骤: Bug 描述规范化 (LLM) <<<")
        # 这一步可能会比较慢，translate_data.py 中已经有进度条
        normalized_df, logs_df, img_logs_df = run_normalization(cleaned_df)
        
        if normalized_df is not None and not normalized_df.empty:
            translation_logs = logs_df
            image_check_logs = img_logs_df
            print("规范化处理完成。")
            dedup_input = normalized_df
        else:
            print("规范化处理未返回有效数据，将使用原始数据进行去重。")
            dedup_input = cleaned_df
    else:
        print("本次清洗未生成有效数据（可能无新数据），尝试查找 cleandata 中最新的文件进行去重...")
        dedup_input = None # dpdata.run_deduplication 会自动处理 None
    
    # 准备额外日志 Sheet
    extra_sheets = {}
    if translation_logs is not None:
        extra_sheets["Translation_Logs"] = translation_logs
    if image_check_logs is not None:
        extra_sheets["Image_Check_Logs"] = image_check_logs

    # 执行去重逻辑 (简单去重 -> 语义去重 -> 聚类)
    # 注意: 现在 run_deduplication 不再直接保存文件，而是返回 DataFrame
    dedup_result = run_deduplication(
        dedup_input, 
        extra_sheets=extra_sheets if extra_sheets else None
    )
    
    if dedup_result["success"]:
        print("\n基础去重与聚类完成！")
        print(f"原始记录数: {dedup_result['original_count']}")
        print(f"剩余记录数: {dedup_result['dedup_count']}")
        print(f"内存传输: {dedup_result['output_path']}")
        
        step2_end = time.time()
        print(f"步骤 2 耗时: {step2_end - step2_start:.2f} 秒")
        
        # 3. AI 语义分析
        # run_ai_analysis 内部会打印自己的步骤标题和耗时
        step3_start = time.time()
        
        df_to_analyze = dedup_result.get("df_final")
        extra_logs_for_ai = dedup_result.get("extra_logs", {})
        
        # 调用 AI 分析，传入内存数据
        ai_result_df, final_save_path = run_ai_analysis(df_input=df_to_analyze, extra_logs=extra_logs_for_ai)
        
        step3_end = time.time()
        
        # 4. 最终兜底去重 (Safety Net Deduplication)
        # 针对 AI 分析后可能仍存在的重复，进行最后一轮简单+语义去重
        if ai_result_df is not None and not ai_result_df.empty:
             print("\n=== 步骤 4: 最终兜底去重 (Safety Net) ===")
             safety_start = time.time()
             original_len = len(ai_result_df)
             
             # 4.1 简单去重 (Subject 完全一致)
             # 注意：AI 可能会把不同的原始 subject 映射到同一个 representative subject，这在 AI 阶段已经处理了(items归并)。
             # 但如果有漏网之鱼，这里再杀一次。
             df_safe = ai_result_df.drop_duplicates(subset=['subject'], keep='first')
             simple_drop = original_len - len(df_safe)
             if simple_drop > 0:
                 print(f"兜底-简单去重: 移除 {simple_drop} 条完全重复数据。")
             
             # 4.2 语义去重 (Difflib, 高阈值 0.85)
             # 这一步是为了防止 AI 分析后，保留的代表条目之间仍然非常相似
             df_safe, sem_drop, _ = semantic_deduplicate(df_safe, threshold=0.85, is_test=False)
             if sem_drop > 0:
                 print(f"兜底-语义去重: 移除 {sem_drop} 条高度相似数据 (阈值 0.85)。")
             
             if simple_drop > 0 or sem_drop > 0:
                 print(f"最终修正: {original_len} -> {len(df_safe)}")
                 # 覆盖保存
                 try:
                     # 读取现有的 Excel 以保留其他 Sheet
                     with pd.ExcelWriter(final_save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                         df_safe.to_excel(writer, sheet_name='Final_Result', index=False)
                     print(f"已更新最终文件: {final_save_path}")
                 except Exception as e:
                     print(f"更新最终文件失败: {e}")
             else:
                 print("兜底检查未发现重复，结果保持不变。")
                 
             safety_end = time.time()
             print(f"步骤 4 耗时: {safety_end - safety_start:.2f} 秒")

    else:
        print(f"\n去重失败或跳过: {dedup_result.get('message')}")
        step2_end = time.time()
        print(f"步骤 2 耗时: {step2_end - step2_start:.2f} 秒")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "=" * 40)
    print(f"       全流程执行结束       ")
    print(f"       总耗时: {total_duration:.2f} 秒       ")
    print("=" * 40)

if __name__ == "__main__":
    main()
