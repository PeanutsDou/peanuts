# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def _script_dir():
    """
    获取当前脚本所在目录的绝对路径
    """
    return os.path.dirname(os.path.abspath(__file__))

def load_config():
    """
    加载项目根目录下的 json/config.json 配置文件
    """
    cfg_path = os.path.join(_script_dir(), "json", "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_latest_dedup_file():
    """
    在 cleandata 目录下查找最新的去重结果 Excel 文件
    排除临时文件和已经经过 AI 分析的文件
    """
    cleandata_dir = os.path.join(_script_dir(), "cleandata")
    if not os.path.isdir(cleandata_dir):
        return None
    files = [
        os.path.join(cleandata_dir, f)
        for f in os.listdir(cleandata_dir)
        if f.endswith('.xlsx') and not f.startswith('~$') and '_AI_Analyzed' not in f
    ]
    if not files:
        return None
    # 按修改时间排序，取最新的一个
    files.sort(key=os.path.getmtime)
    return files[-1]

def analyze_subjects_with_llm(subjects, config, model_name=None):
    """
    调用 LLM (大语言模型) 对一组文本进行精细化语义分析和去重
    
    参数:
        subjects: 文本列表
        config: 配置字典
        model_name: 指定使用的模型名称 (可选)
        
    返回:
        JSON 格式的结构化数据列表: [{ "representative": "...", "items": ["..."] }, ...]
    """
    llm_cfg = config.get("llm_settings", {})
    app_id = llm_cfg.get("app_id", "")
    app_key = llm_cfg.get("app_key", "")
    base_url = llm_cfg.get("base_url", "").rstrip("/")
    
    # 如果指定了 model_name 则使用，否则从配置读取默认 model
    if model_name:
        model = model_name
    else:
        model = llm_cfg.get("model", "glm-4.5-flash")
        
    timeout = llm_cfg.get("timeout", 120)
    max_retries = llm_cfg.get("max_retries", 3)
    
    if not app_key:
        print("错误: 配置文件中未设置 LLM app_key，无法进行 AI 分析。")
        # 降级处理：每个条目都作为独立代表
        return [{"representative": s, "items": [s]} for s in subjects]
        
    # 构造认证头 (适配不同的 API 格式，这里假设是 Bearer Token 模式)
    auth_key = f"{app_id}.{app_key}" if app_id else app_key
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }
    
    # 构造 Prompt 内容
    # 限制每个条目的长度，防止 Token 溢出
    # 将 scene name 前置到 subject 前面，增强 AI 的场景感知
    content_list = []
    for i, s in enumerate(subjects):
        # 尝试从 subjects 原文中找回 scene name (这在之前的逻辑里没有传进来)
        # 这里 subjects 只是 str list，我们可能需要修改调用方 analyze_single_cluster 传入更多信息
        # 或者简单的，我们假设 subjects 里已经是处理过的字符串，或者我们在这里无法获取 scene name
        # 等等，analyze_subjects_with_llm 只接收了 subjects 列表
        
        # 修正方案：
        # 我们需要在 analyze_single_cluster 中，构造传给 LLM 的字符串时，把 scene name 拼进去
        # 但 analyze_subjects_with_llm 的签名是 subjects list
        # 为了不破坏签名，我们可以在 analyze_single_cluster 调用前，把 scene name 拼接到 subject 字符串里
        # 例如: "[主城] 无法移动"
        # 这样 LLM 就能看到了。然后返回的结果也是带 [主城] 的，我们需要在后处理时去掉吗？
        # 提示词里说 "严禁修改"，所以代表条目会带上 [主城]。
        # 这可能会影响后续的匹配 (match_row)，因为 match_row 里的 subject 是不带 [主城] 的。
        
        # 让我们看看 analyze_single_cluster 是怎么调用的
        pass

    content_text = "\n".join([f"{i+1}. {str(s)[:500]}" for i, s in enumerate(subjects)])
    
    prompt = f"""
你是一名资深游戏测试专家。以下是一组经过初步筛选的玩家反馈（可能包含同类问题，也可能混入了无关条目）。
请对这些反馈进行**精细化语义归类**。

**核心原则：**
1. **精准分组**：将描述**同一具体问题**的反馈归为一组。不要把仅是关键词相同但问题不同的反馈强行合并。
2. **场景加权**：如果多条反馈的 **[场景名称]** 相同（如都标记为[主城]或[副本A]），它们描述同一问题的可能性更高，请优先考虑合并。
3. **优选代表**：从每组中选出**信息量最大、表述最清晰、包含关键复现要素**的条目作为“代表条目”(Representative)。
4. **全面覆盖**：输入列表中的**所有条目**都必须被分配到某个组中，不可遗漏。
5. **包容模糊**：将描述笼统（如“卡了”）的条目归入最接近的具体问题组（如“主城传送卡顿”）中；如果无法归类，则单独成组。

**示例参考：**

*案例1 (同类合并 - 笼统与具体)*
输入：
1. [未知场景] 游戏很卡
2. [未知场景] 游戏一卡一卡的
3. [未知场景] 游戏内异常卡顿
4. [未知场景] 游戏存在卡顿问题，且修复延迟超过4天
5. [未知场景] 游戏画面出现闪回现象
输出：
[
  {{ "representative": "[未知场景] 游戏存在卡顿问题，且修复延迟超过4天", "items": ["[未知场景] 游戏很卡", "[未知场景] 游戏一卡一卡的", "[未知场景] 游戏内异常卡顿", "[未知场景] 游戏存在卡顿问题，且修复延迟超过4天"] }},
  {{ "representative": "[未知场景] 游戏画面出现闪回现象", "items": ["[未知场景] 游戏画面出现闪回现象"] }}
]
*(注：通用的“卡顿”归为一组；“闪回”属于特定渲染Bug，虽相关但问题性质不同，故单独列出)*

*案例2 (混合拆分 - 场景与现象不同)*
输入：
1. [迷失乐园] 划艇视角自动转动且划艇可能原地转圈
2. [迷失乐园] 鳄鱼Boss战斗后鸭子船乱转
3. [迷失乐园] 鸭子船在原地打转不前进
4. [迷失乐园] 水变成方块水了，这bug啥时候修，而且鸭子船一直在转圈
5. [迷失乐园] 水面反射不正常，像面镜子一样
6. [高地工厂] 最后到boss前的动画没有水直接摔下来了
7. [未知场景] 水池不会游泳了
输出：
[
  {{ "representative": "[迷失乐园] 划艇视角自动转动且划艇可能原地转圈", "items": ["[迷失乐园] 划艇视角自动转动且划艇可能原地转圈", "[迷失乐园] 鳄鱼Boss战斗后鸭子船乱转", "[迷失乐园] 鸭子船在原地打转不前进"] }},
  {{ "representative": "[迷失乐园] 水变成方块水了，这bug啥时候修，而且鸭子船一直在转圈", "items": ["[迷失乐园] 水变成方块水了，这bug啥时候修，而且鸭子船一直在转圈", "[迷失乐园] 水面反射不正常，像面镜子一样"] }},
  {{ "representative": "[高地工厂] 最后到boss前的动画没有水直接摔下来了", "items": ["[高地工厂] 最后到boss前的动画没有水直接摔下来了"] }},
  {{ "representative": "[未知场景] 水池不会游泳了", "items": ["[未知场景] 水池不会游泳了"] }}
]
*(注：根据【场景】(迷失乐园 vs 高地工厂) 和 【核心现象】(鸭子船转圈 vs 水体渲染异常 vs 动画缺失) 进行拆分。混合描述如条目4，如果同时包含两个问题，可归入其中一个主要组，或根据语境归类)*

*案例3 (层级归纳)*
输入：
1. [全局] 游戏很卡
2. [主城] 进主城特别卡
3. [主城] 每次传送回主城都要卡顿3秒
输出：
[
  {{ "representative": "每次传送回主城都要卡顿3秒", "items": ["[全局] 游戏很卡", "[主城] 进主城特别卡", "[主城] 每次传送回主城都要卡顿3秒"] }}
]

**任务要求：**
请直接返回JSON列表，列表元素包含：
- "representative": 选出的代表条目原文（必须来自输入列表，**严禁修改**）。
- "items": 该组内包含的所有条目原文列表。

**反馈列表：**
{content_text}
"""
    input_chars = len(prompt)
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # 低温度以保证确定性
        "max_tokens": 4000
    }
    
    url = f"{base_url}/chat/completions"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            output_chars = len(content)
            
            # 清理可能存在的 Markdown 代码块标记
            clean_content = re.sub(r'^```json\s*', '', content.strip())
            clean_content = re.sub(r'^```\s*', '', clean_content)
            clean_content = re.sub(r'\s*```$', '', clean_content)
            
            parsed_result = json.loads(clean_content)
            
            # 简单验证返回格式是否符合预期
            if isinstance(parsed_result, list):
                valid = True
                for item in parsed_result:
                    if not isinstance(item, dict) or 'representative' not in item or 'items' not in item:
                        valid = False
                        break
                if valid:
                    return parsed_result, input_chars, output_chars
            
            # 如果格式不对，视为失败，进入重试
            # print(f"LLM 返回格式无效 (尝试 {attempt+1}): {content[:100]}...")
            
        except Exception as e:
            # print(f"LLM 调用失败 (尝试 {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                
    # 如果所有重试都失败，进行降级处理：全部保留，不进行去重
    return [{"representative": s, "items": [s]} for s in subjects], input_chars, 0

def analyze_single_cluster(args):
    """
    处理单个聚类的任务函数
    
    参数 args 包含:
        cid: 聚类 ID
        group: 该聚类的 DataFrame 数据
        config: 配置对象
        threshold: 触发 AI 分析的数量阈值
        cohesion_threshold: 触发 AI 分析的内聚度阈值
        
    返回:
        (cid, cluster_results, cluster_logs, input_chars, output_chars)
    """
    cid, group, config, threshold, cohesion_threshold = args
    
    count = len(group)
    subjects = group['subject'].astype(str).tolist()
    
    # 获取预计算的内聚度 (如果存在)
    cohesion = 0.0
    if 'cohesion' in group.columns and not group['cohesion'].isna().all():
        cohesion = group['cohesion'].iloc[0]
        
    # 判断是否需要进行 AI 分析
    reasons = []
    if count > threshold:
        reasons.append(f"数量{count}>{threshold}")
    if count > 1 and cohesion < cohesion_threshold:
        reasons.append(f"内聚度{cohesion:.2f}<{cohesion_threshold}")
        
    cluster_results = []
    cluster_logs = []
    input_chars = 0
    output_chars = 0
    
    if reasons:
        # 准备带有场景信息的 subject 列表传给 LLM
        subjects_for_llm = []
        # 建立 映射: 带场景的subject -> 原始subject
        # 注意：如果同一个 subject 有不同的场景 (极少见，因为是聚类组内)，可能会有冲突，但组内通常是相似的
        # 为了安全，我们只在传给 LLM 时拼接，返回后再映射回来
        
        # 更好的方法：修改 analyze_subjects_with_llm 接受 (subject, scene_name) 的元组，或者直接在外面拼好
        # 这里我们在外面拼好
        llm_input_map = {} # formatted_subj -> original_subj
        
        for subj in subjects:
            # 找到对应的行
            # 注意：group['subject'] 可能有重复，但这里 subjects 是 list，我们按顺序或值匹配
            # 为了准确获取 scene name，我们需要遍历 group
            # 但 group 可能有多行对应同一个 subject
            rows = group[group['subject'].astype(str) == subj]
            scene_name = ""
            if not rows.empty:
                # 优先取非空的 scene name
                valid_scenes = rows['scene name'].dropna().astype(str)
                valid_scenes = valid_scenes[valid_scenes.str.strip() != ""]
                if not valid_scenes.empty:
                    scene_name = valid_scenes.iloc[0]
            
            if scene_name:
                formatted = f"[{scene_name}] {subj}"
            else:
                formatted = f"[未知场景] {subj}"
            
            subjects_for_llm.append(formatted)
            llm_input_map[formatted] = subj
            
        # 判定使用的模型
        # 逻辑: 小于内聚度阈值 且 聚类数 > 阈值(默认5) -> 使用 review_model (deepseek)
        llm_settings = config.get("llm_settings", {})
        large_cluster_threshold = llm_settings.get("ai_large_cluster_threshold", 5)
        
        target_model = None
        # 注意: reasons 列表是基于 (count > threshold) 和 (count > 1 and cohesion < cohesion_threshold) 生成的
        # 这里我们需要显式判断条件
        is_low_cohesion = (count > 1 and cohesion < cohesion_threshold)
        is_large_cluster = (count > large_cluster_threshold)
        
        if is_low_cohesion and is_large_cluster:
            target_model = llm_settings.get("review_model", "deepseek-v3.2-latest")
            # print(f"  [Cluster {cid}] 触发 DeepSeek 分析 (Count={count}, Cohesion={cohesion:.2f})")
            
        # 满足条件，调用 AI 进行精细分析
        ai_groups, in_c, out_c = analyze_subjects_with_llm(subjects_for_llm, config, model_name=target_model)
        input_chars += in_c
        output_chars += out_c
        
        handled_subjects = set()
        
        for grp in ai_groups:
            rep_formatted = grp.get('representative', '')
            items_formatted = grp.get('items', [])
            
            # 还原回原始 subject
            rep_subj = llm_input_map.get(rep_formatted)
            
            # 如果还原失败 (AI 修改了文本)，尝试模糊匹配
            if not rep_subj:
                # 尝试去掉 [xxx] 前缀
                clean_rep = re.sub(r'^\[.*?\]\s*', "", rep_formatted)
                if clean_rep in subjects:
                    rep_subj = clean_rep
                else:
                    # 还是找不到，可能是 AI 幻觉，或者选取了不存在的。
                    # 兜底：在 input map values 里找最像的，或者跳过
                    # 这里简单处理：如果 items 有效，取 items 第一个作为 rep
                    pass

            # 重新构建 items 列表 (原始 subject)
            real_items = []
            if items_formatted: # 确保 items_formatted 不为空
                for item_fmt in items_formatted:
                    real_item = llm_input_map.get(item_fmt)
                    if not real_item:
                         clean_item = re.sub(r'^\[.*?\]\s*', "", item_fmt)
                         if clean_item in subjects:
                             real_item = clean_item
                    
                    if real_item:
                        real_items.append(real_item)
            
            # 如果 rep_subj 没找到，尝试从 real_items 中恢复
            if not rep_subj and real_items:
                rep_subj = real_items[0]
            
            # 记录详细日志 (在这里使用 items，所以要确保 items 已经解析)
            # 注意：这里的 items 应该是 real_items，即原始 subjects
            # 如果之前的代码直接使用了 items (即 items_formatted)，在循环外可能会导致 items 未定义
            
            if not rep_subj: continue
                
            # --- 以下逻辑使用 rep_subj (原始) 进行后续处理 ---
            
            # 尝试找回对应的图片 URL 和 task_id
            # 优先从代表条目对应的行中获取
            match_row = group[group['subject'].astype(str) == rep_subj]
            img = ""
            task_id = ""
            needs_image = ""
            needs_image_reason = ""
            
            if not match_row.empty:
                row = match_row.iloc[0]
                img = row.get('image', '')
                task_id = row.get('task_id', '')
                needs_image = row.get('needs_image', '')
                needs_image_reason = row.get('needs_image_reason', '')
            else:
                # 如果代表条目没找到图，尝试从同组其他条目中找一张图作为代表
                # 尽量找 needs_image=Yes 的
                found = False
                for item_subj in real_items:
                    m_row = group[group['subject'].astype(str) == item_subj]
                    if not m_row.empty:
                        row = m_row.iloc[0]
                        img_cand = row.get('image', '')
                        if img_cand:
                            img = img_cand
                            task_id = row.get('task_id', '') # 使用有图的这个 task_id? 还是随便一个? 通常代表条目更重要
                            needs_image = row.get('needs_image', '')
                            needs_image_reason = row.get('needs_image_reason', '')
                            found = True
                            break
                
                if not found: # 仍未找到，取组内任意非空图片
                     valid_imgs = [x for x in group['image'].tolist() if pd.notna(x) and str(x).strip()]
                     img = valid_imgs[0] if valid_imgs else ""
                     # 如果没找到图，task_id 还是尽量取 representative 的
                     if not task_id:
                         # 再次尝试匹配 representative (虽然前面没匹配到，可能是类型问题，这里再试一次或取第一个)
                         if not match_row.empty:
                             task_id = match_row.iloc[0].get('task_id', '')
                         else:
                             task_id = group.iloc[0].get('task_id', '')
            
            # 添加保留结果
            res_item = {
                "task_id": task_id,
                "subject": rep_subj,
                "image": img,
                "needs_image": needs_image,
                "needs_image_reason": needs_image_reason,
                "scene": row.get('scene', ''),
                "scene name": row.get('scene name', ''), # 确保 scene name 传递
                "pos": row.get('pos', ''), # 确保 pos 传递
                "cluster_id": cid
            }
            cluster_results.append(res_item)
            
            # 记录详细日志
            for item in real_items:
                handled_subjects.add(item)
                if item == rep_subj:
                    cluster_logs.append({
                        "cluster_id": cid,
                        "original_subject": item,
                        "action": "kept",
                        "reason": "ai_representative",
                        "final_subject": rep_subj
                    })
                else:
                    cluster_logs.append({
                        "cluster_id": cid,
                        "original_subject": item,
                        "action": "removed",
                        "reason": "ai_merged_into_rep",
                        "final_subject": rep_subj
                    })
                    
        # 处理 AI 可能遗漏的条目 (作为兜底，全部保留)
        for s in subjects:
            if s not in handled_subjects:
                 row = group[group['subject'].astype(str) == s].iloc[0]
                 cluster_results.append({
                     "task_id": row.get('task_id', ''),
                     "subject": s,
                     "image": row.get('image', ''),
                     "needs_image": row.get('needs_image', ''),
                     "needs_image_reason": row.get('needs_image_reason', ''),
                     "scene": row.get('scene', ''),
                     "scene name": row.get('scene name', ''), # 确保 scene name 传递
                     "cluster_id": cid
                 })
                 cluster_logs.append({
                     "cluster_id": cid,
                     "original_subject": s,
                     "action": "kept",
                     "reason": "ai_omitted_fallback",
                     "final_subject": s
                 })

    else:
        # 不需要 AI 分析的小簇：直接选取最长文本作为代表
        # 这种方式简单高效，适用于内聚度高且数量少的簇
        representative_row = group.loc[group['subject'].str.len().idxmax()]
        rep_subj = representative_row['subject']
        
        cluster_results.append({
            "task_id": representative_row.get('task_id', ''),
            "subject": rep_subj,
            "image": representative_row.get('image', ''),
            "needs_image": representative_row.get('needs_image', ''),
            "needs_image_reason": representative_row.get('needs_image_reason', ''),
            "scene": representative_row.get('scene', ''),
            "scene name": representative_row.get('scene name', ''), # 确保 scene name 传递
            "pos": representative_row.get('pos', ''),
            "cluster_id": cid
        })
        
        for _, row in group.iterrows():
            orig_subj = row['subject']
            if orig_subj == rep_subj:
                cluster_logs.append({
                    "cluster_id": cid,
                    "original_subject": orig_subj,
                    "action": "kept",
                    "reason": "small_cluster_rep",
                    "final_subject": rep_subj
                })
            else:
                cluster_logs.append({
                    "cluster_id": cid,
                    "original_subject": orig_subj,
                    "action": "removed",
                    "reason": "small_cluster_folded",
                    "final_subject": rep_subj
                })
                
    return cid, cluster_results, cluster_logs, input_chars, output_chars

def run_ai_analysis(df_input=None, extra_logs=None):
    """
    AI 分析阶段主入口函数
    负责加载数据、调度并发任务、整合结果并保存
    :param df_input: 输入的 DataFrame (如果为 None 则尝试从文件加载)
    :param extra_logs: 之前步骤产生的日志 (dict)
    """
    start_time = time.time()
    print("\n=== 步骤 3: AI 语义分析与最终去重 ===")
    
    cfg = load_config()
    llm_settings = cfg.get("llm_settings", {})
    if not llm_settings.get("enabled", False):
        print("提示: 配置文件中 LLM 分析功能未启用 (enabled=false)，跳过此步骤。")
        return None
        
    threshold = llm_settings.get("ai_analyze_threshold", 5)
    cohesion_threshold = llm_settings.get("ai_cohesion_threshold", 0.9)
    
    df = None
    target_file = "Memory_Input"
    existing_sheets = {}
    
    if df_input is not None:
        df = df_input.copy()
        print("已接收内存数据进行分析...")
        if extra_logs and 'extra_sheets' in extra_logs:
             existing_sheets = extra_logs['extra_sheets']
             # cluster_logs 也可能需要
             if 'cluster_logs' in extra_logs and extra_logs['cluster_logs']:
                 existing_sheets['Cluster_Logs'] = pd.DataFrame(extra_logs['cluster_logs'])
    else:
        target_file = find_latest_dedup_file()
        if not target_file:
            print("警告: 未找到可处理的去重结果文件 (cleandata/*.xlsx)。")
            return None
            
        print(f"正在读取文件: {target_file}")
        
        try:
            xls = pd.ExcelFile(target_file)
            df = pd.read_excel(xls, sheet_name='Final_Deduplicated')
            
            for sheet_name in xls.sheet_names:
                if sheet_name != 'Final_Deduplicated':
                    existing_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"读取 Excel 文件失败: {e}")
            return None

    if 'cluster_id' not in df.columns:
        print("错误: 数据中缺少 'cluster_id' 列，无法进行聚类分析。")
        # 尝试自动修复：如果只有少量数据且确实没有聚类，可以全量当做一个簇处理，或者全当噪点
        # 这里选择安全做法：返回原数据，不进行进一步 AI 压缩
        return df, output_path if df_input is None else "Memory_Input"

    final_results = []
    ai_logs = []
    
    # 初始化 Token 统计
    total_input_chars = 0
    total_output_chars = 0
    
    # 1. 处理噪点 (cluster_id == -1) - 这些通常是无法归类的离群点，直接保留
    df_noise = df[df['cluster_id'] == -1]
    for _, row in df_noise.iterrows():
        final_results.append({
            "task_id": row.get('task_id', ''),
            "subject": row['subject'],
            "image": row.get('image', ''),
            "needs_image": row.get('needs_image', ''),
            "scene": row.get('scene', ''),
            "scene name": row.get('scene name', ''), # 确保 scene name 传递
            "pos": row.get('pos', '')
        })
        ai_logs.append({
            "cluster_id": -1,
            "original_subject": row['subject'],
            "action": "kept",
            "reason": "noise",
            "final_subject": row['subject']
        })
        
    # 2. 准备聚类任务
    grouped = df[df['cluster_id'] != -1].groupby('cluster_id')
    groups_to_process = []
    for cid, group in grouped:
        groups_to_process.append((cid, group, cfg, threshold, cohesion_threshold))
        
    print(f"开始分析 {len(groups_to_process)} 个聚类 (使用并发处理)...")
    
    # 初始化进度条
    pbar = None
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(groups_to_process), desc="AI 聚类分析进度", unit="个")
    except ImportError:
        print("提示: 未安装 tqdm 库，将不显示详细进度条。")
    
    max_workers = 5  # 控制并发数，避免对 LLM 接口造成过大压力
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_cluster, args): args[0] for args in groups_to_process}
        
        completed_count = 0
        total_count = len(groups_to_process)
        
        for future in as_completed(futures):
            cid = futures[future]
            try:
                _, c_results, c_logs, in_c, out_c = future.result()
                final_results.extend(c_results)
                ai_logs.extend(c_logs)
                total_input_chars += in_c
                total_output_chars += out_c
            except Exception as e:
                print(f"Cluster {cid} 处理异常: {e}")
            
            if pbar:
                pbar.update(1)
            else:
                completed_count += 1
                if completed_count % 10 == 0:
                    print(f"    进度: {completed_count}/{total_count}")

    if pbar:
        pbar.close()

    # 3. 整理日志与结果
    remove_logs_records = []

    # 3.1 整合旧的移除日志 (如果有)
    existing_remove_df = existing_sheets.get('Removed_Logs')
    if existing_remove_df is not None:
        for _, row in existing_remove_df.iterrows():
            removed_subj = row.get('removed') if 'removed' in existing_remove_df.columns else row.get('subject')
            if pd.isna(removed_subj): continue
            
            remove_logs_records.append({
                "subject": str(removed_subj),
                "stage": str(row.get('stage', 'semantic_dedup')),
                "reason": str(row.get('similarity', '')),
                "representative_subject": str(row.get('kept', ''))
            })

    # 3.2 整合本次 AI 分析的移除日志
    for log in ai_logs:
        if log.get("action") == "removed":
            remove_logs_records.append({
                "subject": str(log.get("original_subject")),
                "stage": "ai_analyze",
                "reason": str(log.get("reason")),
                "representative_subject": str(log.get("final_subject"))
            })

    df_remove_logs = None
    if remove_logs_records:
        df_remove_logs = pd.DataFrame(remove_logs_records)
        # 统一列顺序
        if not df_remove_logs.empty:
            cols_remove = ["subject", "stage", "reason", "representative_subject"]
            # 确保列存在
            for c in cols_remove:
                if c not in df_remove_logs.columns: df_remove_logs[c] = ""
            df_remove_logs = df_remove_logs[cols_remove]

    df_final_output = pd.DataFrame(final_results)
    
    # 调整列顺序: task_id, subject, image, needs_image, scene name, scene, pos
    # 移除 needs_image_reason
    cols_order = ['task_id', 'subject', 'image', 'needs_image', 'scene name', 'scene', 'pos']
    # 确保所有列存在
    for c in cols_order:
        if c not in df_final_output.columns:
            df_final_output[c] = ""
            
    df_final_output = df_final_output[cols_order]
    
    # 生成输出文件名
    if df_input is not None:
         # 如果是内存输入，需要自己构造一个文件名
         timestamp = time.strftime("%Y%m%d")
         cleandata_dir = os.path.join(_script_dir(), "cleandata")
         if not os.path.isdir(cleandata_dir):
            os.makedirs(cleandata_dir, exist_ok=True)
            
         # 查找当前版本
         existing_files = [f for f in os.listdir(cleandata_dir) if f"deduplicated_data_{timestamp}" in f and "_AI_Analyzed" in f]
         version = len(existing_files) + 1
         output_filename = f"deduplicated_data_{timestamp}_v{version}_AI_Analyzed.xlsx"
         output_path = os.path.join(cleandata_dir, output_filename)
    else:
        output_filename = os.path.basename(target_file).replace(".xlsx", "_AI_Analyzed.xlsx")
        output_path = os.path.join(os.path.dirname(target_file), output_filename)
    
    is_test = cfg.get("isTest", False)
    
    # 4. 保存文件 (支持自动重命名以避免占用)
    saved = False
    save_path = output_path
    
    # 准备要保存的数据表
    sheets_to_save = {}
    
    # 提取 Image Check Logs (单独的 Sheet)
    image_check_logs = []
    for item in final_results:
        # 只记录有原因的，或者所有都记录
        if item.get("task_id") or item.get("subject"):
            image_check_logs.append({
                "task_id": item.get("task_id", ""),
                "subject": item.get("subject", ""),
                "needs_image": item.get("needs_image", ""),
                "reason": item.get("needs_image_reason", "")
            })
    
    # 正式模式仅保存最终结果和Image Check Logs (根据用户需求，其他logs不需要展示图片相关的表头，可以删除，这里是指Image_Check_Logs是单独展示图片相关信息的)
    # 实际上，用户说“增设一个单独展示subject是否看图和原因分析的日志表格”，所以无论测试还是正式，都应该有这个表
    sheets_to_save['Final_Result'] = df_final_output
    sheets_to_save['Image_Check_Logs'] = pd.DataFrame(image_check_logs)

    # --- Regenerate Cluster Logs (User Request) ---
    # 重新生成 Cluster Logs，包含：Cluster ID, Count, Cohesion, Representative, Original Subjects, Kept Subjects
    # 移除冗余表头
    new_cluster_logs = []
    
    # 建立 Kept Subjects 映射: cluster_id -> list of kept subjects
    kept_map = {}
    for item in final_results:
        cid = item.get('cluster_id')
        subj = item.get('subject')
        if cid is not None and cid != -1:
             if cid not in kept_map: kept_map[cid] = []
             kept_map[cid].append(str(subj))
             
    if 'cluster_id' in df.columns:
        grouped = df[df['cluster_id'] != -1].groupby('cluster_id')
        for cid, group in grouped:
            # Basic info
            count = len(group)
            cohesion = 0.0
            if 'cohesion' in group.columns and not group['cohesion'].isna().all():
                cohesion = group['cohesion'].iloc[0]
            
            # Original Subjects
            orig_subjs = group['subject'].astype(str).tolist()
            orig_subjs_str = " | ".join(orig_subjs)
            
            # Representative
            # 优先使用保留条目中最长的作为代表
            kept_list = kept_map.get(cid, [])
            kept_subjs_str = " | ".join(kept_list)
            
            if kept_list:
                rep_subj = max(kept_list, key=len)
            else:
                rep_subj = max(orig_subjs, key=len) if orig_subjs else ""
                
            new_cluster_logs.append({
                "Cluster ID": cid,
                "Count": count,
                "Cohesion": round(cohesion, 4),
                "Representative": rep_subj,
                "Original Subjects": orig_subjs_str,
                "Kept Subjects": kept_subjs_str
            })
            
    # Sort by Count desc
    new_cluster_logs.sort(key=lambda x: x["Count"], reverse=True)
    sheets_to_save['Cluster_Logs'] = pd.DataFrame(new_cluster_logs)

    if is_test:
        # 测试模式保存详细的中间日志
        # 用户要求去除 Removed_Logs
        # sheets_to_save['Removed_Logs'] = df_remove_logs if df_remove_logs is not None else pd.DataFrame()
        sheets_to_save['AI_Analysis_Logs'] = pd.DataFrame(ai_logs) if ai_logs else pd.DataFrame()
        
        # 保留原有的其他 sheets (去除 Removed_Logs 和 Cluster_Logs，因为已重新生成)
        for sheet_name, df_sheet in existing_sheets.items():
            if sheet_name not in ['Removed_Logs', 'Image_Check_Logs', 'Cluster_Logs']:
                # 清理其他 sheet 中的图片相关列 (如果存在)
                # "其他日志表格中不需要展示图片相关的表头"
                cols_to_remove = ['image', 'needs_image', 'needs_image_reason', 'Image', 'Needs Image', 'Reason']
                existing_cols = df_sheet.columns
                valid_cols = [c for c in existing_cols if c not in cols_to_remove]
                sheets_to_save[sheet_name] = df_sheet[valid_cols]

    try:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for name, df_s in sheets_to_save.items():
                df_s.to_excel(writer, sheet_name=name, index=False)
        saved = True
    except PermissionError:
        print(f"警告: 文件 {save_path} 正被其他程序占用，尝试保存为副本...")
        save_path = save_path.replace(".xlsx", f"_{int(time.time())}.xlsx")
        try:
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                for name, df_s in sheets_to_save.items():
                    df_s.to_excel(writer, sheet_name=name, index=False)
            saved = True
        except Exception as e:
            print(f"错误: 保存副本失败: {e}")
    except Exception as e:
        print(f"错误: 保存文件失败: {e}")
            
    end_time = time.time()
    duration = end_time - start_time
    
    # 统计信息
    original_count = len(df)
    final_count = len(df_final_output)
    retention_rate = (final_count / original_count * 100) if original_count > 0 else 0
    
    print("-" * 30)
    print(f"AI 分析阶段完成")
    print(f"耗时: {duration:.2f} 秒")
    print(f"原始条目: {original_count}")
    print(f"最终条目: {final_count}")
    print(f"留存比例: {retention_rate:.2f}%")
    print(f"Token 消耗参考: 输入字数 {total_input_chars}, 输出字数 {total_output_chars}")
    print(f"结果文件已保存至: {save_path}")
    print("-" * 30)
    
    return df_final_output, save_path

if __name__ == "__main__":
    run_ai_analysis()
