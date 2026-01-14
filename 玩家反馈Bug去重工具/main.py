# -*- coding: utf-8 -*-
from clean_data import run
from dpdata import run_deduplication
from AI_analyze_data import run_ai_analysis
import sys
import pandas as pd

def main():
    # 1. 执行数据清洗
    print("=== 步骤 1: 数据清洗 ===")
    result = run()
    
    cleaned_df = None
    
    if result["success"]:
        # 如果有 message，说明虽然成功但可能没有新数据
        if "message" in result:
            print("\n提示: {}".format(result['message']))
        else:
            print("\n数据处理成功！")
            if "new_count" in result:
                print("本次新增清洗数据: {} 条".format(result['new_count']))
            if "cleaned_df" in result:
                cleaned_df = result['cleaned_df']
                print("清洗后数据已加载到内存，准备去重...")
                
        # 仍然打印源文件信息
        if "data" in result and result["data"]:
            print("\n处理的源文件列表:")
            for it in result["data"]:
                print('- {} ({})'.format(it["name"], it["type"]))
    else:
        print("\n清洗失败: {}".format(result.get('message', '未知错误')))
        return
        
    print('\n原始数据目录: {}'.format(result.get("rawdata_dir")))
    print('配置文件路径: {}'.format(result.get("config_path")))
    
    # 2. 执行数据去重
    print("\n=== 步骤 2: 数据去重 ===")
    
    dedup_input = None
    
    if cleaned_df is not None and not cleaned_df.empty:
        # 优先使用内存中的数据
        dedup_input = cleaned_df
    else:
        print("本次清洗未生成有效数据（可能无新数据），尝试查找 cleandata 中最新的文件进行去重...")
        dedup_input = None # run_deduplication handles None by finding latest file
    
    dedup_result = run_deduplication(dedup_input)
    
    if dedup_result["success"]:
        print("\n去重完成！")
        print("原始记录数: {}".format(dedup_result["original_count"]))
        print("剩余记录数: {}".format(dedup_result["dedup_count"]))
        print("最终文件路径: {}".format(dedup_result["output_path"]))
        
        # 3. AI 语义分析
        run_ai_analysis()
        
    else:
        print("\n去重失败/跳过: {}".format(dedup_result.get("message")))

if __name__ == "__main__":
    main()
