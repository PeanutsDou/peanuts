# -*- coding: utf-8 -*-
import os
import pandas as pd
import sys
import time

# Python 2/3 compatibility
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except NameError:
    pass

# Try to import LLMClient
try:
    from llm_client import LLMClient
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False
    print("Warning: llm_client module not found.")

def safe_print(msg):
    """安全打印函数，处理编码问题"""
    try:
        print(msg)
    except:
        try:
            print(msg.encode('utf-8'))
        except:
            pass

def sort_and_group(df, subject_col="subject", model=None, target_group_size=15, max_clusters=30):
    """
    使用 LLM (Chat Mode) 对数据进行深度语义分类
    
    Args:
        df: 包含 bug 描述的 DataFrame
        subject_col: 包含文本内容的列名
        model: 这里复用参数名，实际可以传入 LLMClient 实例
        target_group_size: (未使用，保留接口兼容)
        max_clusters: (未使用，保留接口兼容)
        
    Returns:
        grouped_data: 一个字典，Key 是聚类ID/名称，Value 是该组的 DataFrame
        df_with_topic: 增加了 'core_topic' 和 'cluster_id' 列的原始 DataFrame
    """
    if df is None or df.empty:
        return {}, df
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    if not HAS_LLM_CLIENT:
        safe_print("未找到 llm_client，无法进行 LLM 分类。")
        df["core_topic"] = "Unsorted"
        df["cluster_id"] = 0
        return {"Unsorted": df}, df

    safe_print(u"\n[分类阶段] 正在使用 GLM-4.5-Flash 进行深度语义分类 (Chat模式)...")
    
    subjects = df[subject_col].tolist()
    
    # 1. 获取/初始化 Client
    client = model
    if client is None or not isinstance(client, LLMClient):
        try:
            client = LLMClient()
            safe_print(u"LLMClient 初始化成功。")
        except Exception as e:
            safe_print(u"LLMClient 初始化失败: {}".format(e))
            df["core_topic"] = "Error"
            df["cluster_id"] = 0
            return {"Error": df}, df

    # 2. 执行分类
    # 为了防止 API 超时或出错导致全盘失败，我们分批处理，但 classify_items 内部已经实现了分批
    safe_print(u"正在对 {} 条数据进行分类...".format(len(subjects)))
    start_time = time.time()
    
    category_map = client.classify_items(subjects, batch_size=10)
    
    end_time = time.time()
    safe_print(u"分类耗时: {:.2f} 秒".format(end_time - start_time))
    
    # 3. 回填结果
    # map 可能会有缺失值 (如果 key 不在 map 中)，填充为 "Unclassified"
    df["core_topic"] = df[subject_col].map(category_map).fillna("Unclassified")
    
    # 4. 生成 cluster_id (根据 core_topic 排序后生成)
    # 按照 topic 名称排序，保证相同 topic 的 ID 一致
    # 也可以按照 topic 出现频率排序
    topic_counts = df["core_topic"].value_counts()
    sorted_topics = topic_counts.index.tolist()
    topic_to_id = {topic: i+1 for i, topic in enumerate(sorted_topics)}
    
    df["cluster_id"] = df["core_topic"].map(topic_to_id)
    
    # 5. 分组返回
    grouped_data = {}
    for topic, group in df.groupby("core_topic"):
        grouped_data[topic] = group
        
    safe_print(u"分类完成，共生成 {} 个语义类别。".format(len(grouped_data)))
    
    # 打印前几个类别示例
    count = 0
    for topic, group in grouped_data.items():
        if count < 5:
            safe_print(u"  - {} ({} 条)".format(topic, len(group)))
            count += 1
            
    return grouped_data, df

if __name__ == "__main__":
    # 测试代码
    safe_print(u"=== 测试 sort_data (LLM Chat Classification) ===")
    
    test_data = [
        {"subject": "登录游戏时发生闪退", "id": 1},
        {"subject": "进入副本后卡死", "id": 2},
        {"subject": "无法登录，提示网络错误", "id": 3},
        {"subject": "充值界面打不开", "id": 4},
        {"subject": "闪退严重", "id": 5},
        {"subject": "背包满了无法拾取", "id": 6},
        {"subject": "登录失败", "id": 7},
        {"subject": "充值没到账", "id": 8},
        {"subject": "副本掉线", "id": 9}
    ]
    
    df_test = pd.DataFrame(test_data)
    groups, df_result = sort_and_group(df_test, "subject")
    
    safe_print(u"\n分类结果:")
    if not df_result.empty:
        for index, row in df_result.iterrows():
            safe_print(u"ID: {}, Topic: {}, Subject: {}".format(row['id'], row['core_topic'], row['subject']))
