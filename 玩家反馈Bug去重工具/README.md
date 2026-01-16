# 玩家反馈Bug去重工具

## 工具简介

本工具旨在自动化处理玩家反馈的Bug和建议数据，通过多阶段的智能处理流程，实现数据的清洗、规范化、去重和语义聚类，最终输出精简、高质量的反馈列表。

## 主要功能与流程详解

工具执行过程严格按照以下流水线进行，每个步骤都经过精心设计以确保数据质量：

### 1. 数据清洗 (Data Cleaning)
*   **对应脚本**: `clean_data.py`
*   **输入**: 读取 `rawdata` 目录下的所有 Excel/CSV 文件。
*   **处理逻辑**: 
    *   自动合并多个源文件。
    *   清理空行和无效数据。
    *   根据 `config.json` 中的 `column_mapping` 提取关键列（`subject`, `image`, `task_id` 等），去除无关的冗余列。
    *   **内存传递**: 清洗后的数据直接以 DataFrame 形式传递给下一步，不再生成中间临时文件，减少磁盘I/O。

### 2. 智能预处理 (LLM Preprocessing)
*   **对应脚本**: `translate_data.py`
*   **断点续传机制 (Checkpoint)**: 
    *   在调用昂贵的 LLM API 之前，先检查 `cache/normalization_checkpoint.json`。
    *   **增量更新**: 仅对**新增**的 Bug 描述进行 API 调用；已分析过的历史数据直接从本地缓存加载，显著节省时间和 Token 成本。
*   **Bug描述规范化**: 
    *   利用 LLM 将口语化的玩家描述（如“这怪打不了”）重写为标准化的 Bug 描述（如“副本X中怪物Y无法被攻击”）。
    *   **语义拆分**: 如果一条反馈包含多个独立的问题，AI 会将其拆分为多条独立的记录。
    *   **结构化提取**: 提取核心主体（Subject Core）、发生位置（Location）和具体现象（Phenomenon）。
*   **图片必要性检测**: 
    *   **对应脚本**: `image_check.py`
    *   并发分析 Bug 描述，智能判断开发人员是否需要查看截图来辅助定位问题。
    *   输出结果包含 `needs_image` (Yes/No) 和 `reason` (判断理由)。

### 3. 多级去重与聚类 (Deduplication & Clustering)
*   **对应脚本**: `dpdata.py`, `Emb_data.py`, `kmeans_data.py`, `DBS_data.py`
*   **基础去重**: 移除文本完全一致的重复条目。
*   **语义去重 (Semantic Deduplication)**: 
    *   使用 `difflib` 算法计算文本相似度。
    *   当相似度超过阈值（默认 0.75）时，视为重复并移除。
*   **向量化与聚类流程**: 
    1.  **向量化 (Vectorization)**:
        *   **对应脚本**: `Emb_data.py`
        *   使用 `m3e-base` 等 Embedding 模型将文本转化为高维向量。
    2.  **K-Means 粗略分区 (Pre-clustering)**:
        *   **对应脚本**: `kmeans_data.py`
        *   **目的**: 解决大数据量直接聚类时的性能瓶颈和密度问题。通过 K-Means 将全量向量划分为多个“粗略批次”（默认 10 个，或根据目标批次大小自动计算）。
        *   **逻辑**: 先将数据在语义空间上粗分，确保后续的高精聚类在相对独立的语义块中进行。
    3.  **HDBSCAN 精细聚类 (Density-Based Clustering)**:
        *   **对应脚本**: `DBS_data.py`, `dpdata.py`
        *   **分批处理**: 对每个 K-Means 批次的数据分别应用 **HDBSCAN** 算法。
        *   **逻辑**: 自动将语义相近的反馈归为一组（Cluster）。
        *   **DBSCAN 优化**: 对 HDBSCAN 产生的噪点数据进行二次扫描，尽可能找回漏网之鱼。

### 4. AI 深度语义分析 (AI Analysis)
*   **对应脚本**: `AI_analyze_data.py`
*   **输入**: 上一步生成的聚类组（Cluster Groups）。
*   **处理逻辑**: 
    *   对每个聚类组调用 LLM 进行精细化语义比对。
    *   **代表选取**: 从每组中选出描述最完整、信息量最大的一条作为“代表条目”（Representative）。
    *   **冗余剔除**: 组内其他条目视为重复，被合并到代表条目下。
    *   **并发处理**: 采用多线程并发调用 AI 接口，大幅提升处理速度。

### 5. 最终兜底去重 (Safety Net)
*   **对应脚本**: `AI_analyze_data.py` (集成在分析流程末尾)
*   **目的**: 解决 AI 分析后可能仍存在的少量重复问题（如 AI 幻觉或漏判）。
*   **逻辑**: 
    *   再次进行全局简单去重。
    *   使用高阈值（0.85+）的语义相似度检查，移除极度相似的残留条目。

## 输出结果

流程结束后，会在 `cleandata` 目录下生成唯一的最终结果文件：
`deduplicated_data_YYYYMMDD_vX_AI_Analyzed.xlsx`

该 Excel 文件包含以下 Sheet：

1.  **Final_Result** (核心结果):
    *   最终去重后的精简列表。
    *   包含列：`task_id`, `subject` (规范化后的描述), `image` (图片链接), `needs_image` (是否需看图)。
2.  **Cluster_Logs**: 记录聚类过程中的合并详情。
3.  **Translation_Logs**: 记录原始描述与 AI 规范化描述的对照。
4.  **Image_Check_Logs**: 记录每条数据的图片检测结果和原因。
5.  **AI_Analysis_Logs**: 记录 AI 深度分析阶段的合并操作日志。

## 环境配置

本工具需要 Python 3.8 或以上版本。

### 1. 安装依赖

请在项目根目录下运行以下命令安装所需依赖库：

```bash
pip install -r requirements.txt
```

**依赖列表**：
*   `pandas`: 数据处理
*   `openpyxl`: Excel读写
*   `requests`: 网络请求（用于调用LLM接口）
*   `tqdm`: 进度条显示
*   `numpy`: 数值计算
*   `scikit-learn`: 机器学习算法（聚类）
*   `sentence-transformers`: 文本向量化模型

### 2. 配置文件

在使用前，请确保 `json/config.json` 已正确配置。主要配置项包括：

*   **rawdata_dir_name**: 原始数据文件夹名称（默认为 `rawdata`）。
*   **cleandata_dir_name**: 结果输出文件夹名称（默认为 `cleandata`）。
*   **llm_settings**: LLM 接口配置（用于规范化和AI分析），需填写 `app_key` 等信息。
*   **deduplication_thresholds**: 去重阈值设置。

## 使用说明

### 运行主流程

最简单的使用方式是直接运行 `main.py` 脚本，它会自动串联所有步骤：

```bash
py -3 main.py
```
*(注：如果您的环境中同时存在Python 2和3，请使用 `py -3` 确保调用Python 3解释器)*

### 运行流程详解

工具执行过程分为以下几个主要步骤，终端会实时显示当前阶段和进度：

1.  **步骤 1: 数据清洗 (Data Cleaning)**
    *   读取 `rawdata` 目录下的所有 Excel/CSV 文件。
    *   合并数据，清理空行，保留关键列（如 `subject`, `image` 等）。
    *   结果暂存于内存或 `cleandata` 目录。

2.  **步骤 2: 数据去重 (Deduplication)**
    *   **规范化 (可选)**：如果配置了 LLM，会先对 Bug 描述进行重写，使其更规范。
    *   **基础去重**：移除完全相同的文本。
    *   **语义去重**：计算文本相似度，移除高度相似的条目。
    *   **向量化与聚类**：将文本转化为向量，使用 HDBSCAN 算法进行聚类，识别潜在的重复组。

3.  **步骤 3: AI 语义分析与最终去重**
    *   对每个聚类组调用 LLM 进行精细化分析。
    *   从每组中选出一个“代表条目”，其余视为重复并移除。
    *   最终结果保存为 `cleandata/deduplicated_data_YYYYMMDD_vX_AI_Analyzed.xlsx`。

## 输出结果

最终生成的 Excel 文件包含以下主要 Sheet：

*   **Final_Result**: 最终去重后的保留数据，包含 `subject` (Bug描述) 和 `image` (图片链接) 等。
*   **Removed_Logs** (测试模式): 记录被移除的条目及其原因（如“语义重复”、“AI合并”等）。
*   **AI_Analysis_Logs** (测试模式): 记录 AI 分析的具体操作日志。

在终端运行结束后，会显示本次处理的统计信息，包括：
*   各阶段耗时
*   总耗时
*   原始条目数 vs 最终条目数
*   留存比例

## 目录结构说明

*   `rawdata/`: 存放待处理的原始数据文件。
*   `cleandata/`: 存放处理后的中间结果和最终结果文件。
*   `json/`: 存放配置文件和缓存数据。
*   `embeddings/`: 存放生成的向量缓存文件（加快后续运行速度）。
*   `cache/`: 存放临时计算文件（如批处理聚类时的临时向量）。
*   `*.py`: 工具的核心脚本代码。

## 更新日志 (2026-01-15)

1.  **临时文件优化**: 聚类过程中的临时 `npy` 文件现在生成于 `cache/` 目录，不再污染根目录。
2.  **字段提取增强**: 
    *   针对 `QData` 类型，自动从“日志原文”中提取 `scene` (场景ID) 和 `pos` (坐标，优先3D坐标)。
    *   针对 `BugFeedback` 类型，从“描述”中提取“问题出现场景”作为 `scene`。
3.  **Task ID 映射**: 每次清洗都会在 `rawdata/` 下生成 `task_id_mapping.xlsx`，用于记录 `task_id` 与源文件的对应关系。
5.  **LLM 提示词优化**: 
    *   更新了 `translate_data.py` 中的提示词，采用了更精简的原则和基于真实数据的典型案例（包含情绪化去噪、简单描述规范化、多问题拆分），以提高规范化和结构化提取的准确性。
    *   移除了旧的 Checkpoint 缓存，强制重新生成规范化数据。

---
**注意**：请勿随意删除 `json/config.json` 文件，否则工具将无法正常运行。
