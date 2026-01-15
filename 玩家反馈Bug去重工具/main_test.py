# -*- coding: utf-8 -*-
from clean_data import run as run_cleaning
from dpdata import run_deduplication, semantic_deduplicate
from AI_analyze_data import run_ai_analysis
from translate_data import run_normalization
import sys
import pandas as pd
import time
import os

def load_from_excel(path):
    """
    加载指定的 Excel 文件作为清洗后的数据
    """
    try:
        df = pd.read_excel(path)
        # 简单检查列名
        if 'subject' not in df.columns:
            print("错误: 指定的表格缺少 'subject' 列。")
            return None
        return df
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def main_test():
    """
    测试版主程序入口
    支持多种测试模式
    """
    total_start_time = time.time()
    print("=" * 40)
    print("    玩家反馈Bug去重工具 - 测试模式启动    ")
    print("=" * 40)
    
    print("\n请选择测试模式:")
    print("1. 随机抽样测试 (输入数量，从全部数据中随机抽取)")
    print("2. 固定采样测试 (输入Excel路径，直接使用该文件数据)")
    print("3. 关键字采样测试 (输入关键词，筛选包含关键词的条目)")
    
    mode = input("\n请输入模式编号 (1/2/3): ").strip()
    
    # 询问是否跳过 AI 预处理步骤
    skip_norm = False
    skip_img = True # 默认跳过图片检查
    
    print("\n[AI 预处理配置]")
    # 语义规范化
    norm_choice = input("是否跳过 [语义规范化] (y/n, 默认 n): ").strip().lower()
    if norm_choice == 'y':
        skip_norm = True
        print("-> 已设置: 跳过语义规范化")
    else:
        print("-> 已设置: 执行语义规范化 (默认)")
        
    # 图片分析
    img_choice = input("是否跳过 [图片查看分析] (y/n, 默认 y): ").strip().lower()
    if img_choice == 'n':
        skip_img = False
        print("-> 已设置: 执行图片查看分析")
    else:
        print("-> 已设置: 跳过图片查看分析 (默认)")

    cleaned_df = None
    
    # --- 数据准备阶段 ---
    
    if mode == '1':
        # 模式 1: 随机抽样
        limit = None
        try:
            user_input = input("请输入采样数量 (直接回车默认100): ").strip()
            if not user_input:
                limit = 100
            elif user_input.isdigit():
                limit = int(user_input)
            else:
                print("输入无效，使用默认值 100。")
                limit = 100
        except:
            limit = 100
            
        print(f"\n正在执行数据清洗并抽取 {limit} 条数据...")
        cleaning_result = run_cleaning()
        
        if cleaning_result["success"] and "cleaned_df" in cleaning_result:
            df = cleaning_result['cleaned_df']
            if df is not None and not df.empty:
                if len(df) > limit:
                    cleaned_df = df.sample(n=limit, random_state=42)
                    print(f"随机抽样完成: {len(df)} -> {len(cleaned_df)}")
                else:
                    cleaned_df = df
                    print(f"数据量 ({len(df)}) 少于采样数 ({limit})，使用全部数据。")
        else:
            print(f"数据清洗失败: {cleaning_result.get('message')}")
            return

    elif mode == '2':
        # 模式 2: 固定文件
        file_path = input("请输入测试用 Excel 文件的绝对路径: ").strip()
        # 去除可能存在的引号
        file_path = file_path.strip('"').strip("'")
        
        if os.path.isfile(file_path):
            print(f"\n正在读取文件: {file_path}")
            cleaned_df = load_from_excel(file_path)
            if cleaned_df is None:
                return
            print(f"成功加载 {len(cleaned_df)} 条数据。")
        else:
            print("文件不存在，请检查路径。")
            return

    elif mode == '3':
        # 模式 3: 关键字筛选
        keywords_input = input("请输入关键词 (多个关键词用逗号隔开): ").strip()
        keywords = [k.strip() for k in keywords_input.replace('，', ',').split(',') if k.strip()]
        
        if not keywords:
            print("未输入有效关键词，退出。")
            return
            
        print(f"\n正在执行数据清洗并筛选包含 {keywords} 的条目...")
        cleaning_result = run_cleaning()
        
        if cleaning_result["success"] and "cleaned_df" in cleaning_result:
            df = cleaning_result['cleaned_df']
            if df is not None and not df.empty:
                # 筛选逻辑: subject 包含任意一个关键词
                # 将 subject 转为字符串，处理 NaN
                mask = df['subject'].astype(str).apply(lambda x: any(k in x for k in keywords))
                cleaned_df = df[mask]
                print(f"筛选完成: 原始 {len(df)} -> 命中 {len(cleaned_df)}")
                if cleaned_df.empty:
                    print("未找到包含指定关键词的条目。")
                    return
        else:
            print(f"数据清洗失败: {cleaning_result.get('message')}")
            return
            
    else:
        print("无效的模式选择，退出。")
        return

    # --- 后续流程 (同 main.py) ---
    
    if cleaned_df is None or cleaned_df.empty:
        print("没有可处理的数据，流程结束。")
        return

    # 2. 执行数据去重 (包含规范化)
    print("\n" + "=" * 30)
    print("步骤 2: 数据去重 (Deduplication)")
    print("=" * 30)
    step2_start = time.time()
    
    dedup_input = None
    translation_logs = None
    keyword_dedup_logs = None
    image_check_logs = None
    
    # 插入步骤：规范化处理
    print("\n>>> 子步骤: Bug 描述规范化 (LLM) <<<")
    normalized_df, logs_df, img_logs_df = run_normalization(cleaned_df, skip_norm=skip_norm, skip_img=skip_img)
    
    if normalized_df is not None and not normalized_df.empty:
        translation_logs = logs_df
        image_check_logs = img_logs_df
        print("规范化处理完成。")
        dedup_input = normalized_df
    else:
        print("规范化处理未返回有效数据，将使用原始数据进行去重。")
        dedup_input = cleaned_df
    
    # 准备额外日志 Sheet
    extra_sheets = {}
    if translation_logs is not None:
        extra_sheets["Translation_Logs"] = translation_logs
    if image_check_logs is not None:
        extra_sheets["Image_Check_Logs"] = image_check_logs
    
    # 执行去重逻辑 (简单去重 -> 语义去重 -> 聚类)
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
             df_safe = ai_result_df.drop_duplicates(subset=['subject'], keep='first')
             simple_drop = original_len - len(df_safe)
             if simple_drop > 0:
                 print(f"兜底-简单去重: 移除 {simple_drop} 条完全重复数据。")
             
             # 4.2 语义去重 (Difflib, 高阈值 0.85)
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
    print(f"       测试流程执行结束       ")
    print(f"       总耗时: {total_duration:.2f} 秒       ")
    print("=" * 40)

if __name__ == "__main__":
    main_test()
