# AstrBot 自主学习插件

## 🚀 项目概述

# 功能尚未完善！开发中！请勿使用

---

AstrBot 自主学习插件是一个为 AstrBot 框架设计的智能对话风格学习与人格优化解决方案。它通过实时消息捕获、多维度数据分析、渐进式学习机制和动态人格更新，使聊天机器人能够模仿特定用户的对话风格，实现更自然、个性化的交互。本插件旨在提供一个功能完善、可扩展且高效的智能体风格学习平台。

## 🔐 **<u>后台管理使用教程</u>**

### **⚠️ <u>重要安全提醒</u>**

**<u>插件启动后请立即访问后台管理页面并修改默认密码！</u>**

### 🌐 访问后台管理

1. **启动插件后**，Web管理界面将在以下地址启动：

   ```
   http://localhost:7833 或 http://你的服务器IP:7833
   ```

2. **首次登录**：

   - 默认密码：`self_learning_pwd`
   - **<u>⚠️ 强烈建议：首次登录后立即修改密码！</u>**

### 🔑 登录流程

1. 浏览器访问管理页面地址
2. 输入默认密码 `self_learning_pwd`
3. **系统将强制要求修改密码**
4. 设置新密码后即可正常使用

### 🛡️ 安全说明

- **<u>请务必在生产环境中修改默认密码！</u>**

## 🏗️ 技术架构与核心逻辑

本插件采用模块化、异步化的架构设计，核心功能围绕 `core` 和 `services` 目录组织，并通过清晰的接口和数据流实现各组件的协同工作。

### 核心服务层 (`services/`)

`services` 目录包含了插件的主要业务逻辑和功能模块，各服务之间通过依赖注入和接口定义进行交互，实现高内聚、低耦合。

-   `message_collector.py`: **消息收集服务**。负责异步捕获 AstrBot 接收到的消息（包括群聊和私聊），进行初步清洗（去重、长度过滤），并缓存消息以实现批量写入优化。它将原始消息和筛选后的消息持久化到**全局消息数据库**。
-   `database_manager.py`: **数据库管理器**。统一管理插件的所有数据持久化。它维护一个**全局消息数据库** (`messages.db`) 用于存储原始消息、筛选消息和学习批次记录。同时，它支持为每个群聊创建独立的 SQLite 数据库 (`group_databases/`)，用于存储用户画像、社交关系、风格档案和人格备份。
-   `multidimensional_analyzer.py`: **多维度分析器**。利用 LLM 对消息进行智能筛选和多维度量化评分。它构建用户画像（活动模式、沟通风格、话题偏好、情感倾向）和情境模式。**社交关系网络功能**：该模块深入分析消息中的 `@关系`、`回复关系` 和 `互动强度`，构建并维护群聊内部的社交关系网络。这些关系数据存储在 `group_databases/` 中的独立 SQLite 数据库内，为后续的人格优化和智能回复提供上下文支持。该服务初始化并使用独立的 LLM 客户端（弱模型用于筛选，强模型用于深度分析）。
-   `style_analyzer.py`: **对话风格分析服务**。专注于对话风格的深度分析，主要利用强 LLM 模型提取数值化的风格档案（如词汇丰富度、句式复杂度、情感表达度等）。它能够检测风格演化并存储历史记录，并基于当前风格和目标人格生成风格优化建议。
-   `learning_quality_monitor.py`: **学习质量监控系统**。评估学习过程的质量，确保人格更新的一致性和稳定性。它通过比较更新前后的人格、分析消息质量分数来判断学习效果，并能检测潜在的质量问题，甚至在质量不达标时暂停学习。
-   `progressive_learning.py`: **渐进式学习协调器**。作为整个学习流程的中央协调者，它整合了上述所有服务。它管理学习会话，控制学习的启动、停止和周期性循环。其核心逻辑是执行学习批次：获取消息 -> 多维度筛选 -> 风格分析 -> 质量评估 -> 应用更新（如果质量达标） -> 标记消息已处理。
-   `intelligent_responder.py`: **智能回复生成器**。基于学习到的人格和上下文信息，结合配置的回复概率，生成并发送个性化回复。
-   `ml_analyzer.py`: **轻量级机器学习分析器**。提供额外的机器学习能力，如用户行为分析、消息聚类和回复质量预测。它支持可选的 `scikit-learn` 集成，并在 LLM 不可用时提供备用算法。
-   `persona_manager.py`: **人格管理器**。负责管理和维护机器人的人格配置，包括获取当前人格描述。
-   `persona_updater.py`: **人格更新器**。负责执行具体的人格更新逻辑，根据风格分析结果和筛选过的消息，动态调整 AstrBot 的默认人格，并管理优质对话样本。
-   `persona_backup_manager.py`: **人格备份管理器**。实现人格配置的自动备份与恢复机制，确保人格数据安全。

### 核心组件 (`core/`)

`core` 目录包含插件的基础设施、通用接口和设计模式，为服务层提供支撑。

-   `interfaces.py`: 定义了插件内部各模块间的抽象接口 (ABC) 和协议 (Protocol)，如 `IMessageCollector`, `IStyleAnalyzer`, `IAsyncService` 等，确保模块间的松耦合和可测试性。同时定义了 `MessageData`, `AnalysisResult` 等标准化数据结构和各种枚举、异常类型。
-   `llm_client.py`: 封装了与外部大型语言模型 (LLM) API 的交互逻辑。它支持通过配置动态切换模型和 API 端点（如 `gpt-4o`, `gpt-4o-mini`），使用 `aiohttp.ClientSession` 进行异步 HTTP 请求，并包含健壮的错误处理机制。
-   `factory.py`: 实现了工厂模式 (`FactoryManager`, `ServiceFactory`, `ComponentFactory`)，用于动态创建和管理服务及内部组件的实例，便于依赖注入和模块化管理。
-   `patterns.py`: 包含常用的设计模式实现，如单例模式等，用于构建可维护和可扩展的代码。

### 整体逻辑流与数据流

1.  **初始化**: 插件启动时，`main.py` 中的 `SelfLearningPlugin` 通过 `FactoryManager` 初始化所有核心服务和内部组件。
2.  **消息监听与收集**: `on_message` 方法监听 AstrBot 的所有消息事件。`QQFilter` 进行 QQ 号过滤，`MessageCollectorService` 将符合条件的消息收集到内存缓存。
3.  **批量持久化**: `MessageCollectorService` 定期将缓存中的消息批量刷新到**全局消息数据库**的 `raw_messages` 表。
4.  **学习周期触发**: `LearningScheduler` 周期性触发 `ProgressiveLearningService` 执行学习批次。
5.  **消息筛选与评分**: `ProgressiveLearningService` 从 `raw_messages` 获取未处理消息。`MultidimensionalAnalyzer` 使用弱 LLM 模型对消息进行智能筛选，并使用强 LLM 模型进行多维度量化评分（内容质量、相关性、情感等），将筛选后的消息连同评分存储到 `filtered_messages` 表。
6.  **风格分析**: `StyleAnalyzerService` 使用强 LLM 模型对筛选后的消息进行深度风格分析，生成风格档案。
7.  **人格更新与质量监控**: `PersonaUpdater` 根据风格分析结果动态更新 AstrBot 的人格设置。`LearningQualityMonitor` 评估更新效果，确保人格的一致性和稳定性。
8.  **记忆重放与强化学习**: `MLAnalyzer` 执行记忆重放等操作，可能涉及强化学习以巩固学习效果。
9.  **数据管理**: `DatabaseManager` 负责所有数据的存储、查询、统计和备份。用户画像、社交关系、风格档案和人格备份存储在**分群数据库**中。
10. **智能回复**: `IntelligentResponder` 根据最新的人格和上下文，生成并发送回复。
11. **命令交互**: 插件提供 `/learning_status`, `/start_learning` 等命令，允许用户查询状态、控制学习流程和管理数据。

## 🛠️ 技术栈

本插件主要基于 **Python 3.8+** 开发，并广泛利用了现代异步编程范式和大型语言模型技术。

-   **编程语言**: Python
-   **异步框架**: `asyncio`
-   **HTTP 客户端**: `aiohttp` (用于异步网络请求，特别是 LLM API 调用)
-   **数据库**: `sqlite3` (内置), `aiosqlite` (异步 SQLite 驱动), `sqlalchemy` (ORM，可能用于更复杂的数据库抽象)
-   **数据结构与验证**: `dataclasses`, `pydantic` (用于配置和数据模型的定义与验证)
-   **机器学习**: `scikit-learn` (可选，用于轻量级 ML 分析), `numpy`, `pandas` (数据处理)
-   **自然语言处理 (NLP)**: `jieba` (中文分词), `nltk`, `spacy` (可能用于更高级的文本处理)
-   **大型语言模型 (LLM)**: 集成外部 LLM API (如 OpenAI GPT 系列，通过 `LLMClient` 灵活配置 `api_url`, `api_key`, `model_name`)
-   **日志**: `loguru`, `structlog` (提供结构化和增强的日志功能)
-   **配置管理**: `pyyaml`, `toml`, `configparser`
-   **测试**: `pytest`, `pytest-asyncio`
-   **其他**: `cachetools` (缓存), `redis` (可选，用于分布式缓存), `pillow`, `opencv-python` (图像处理，如果消息包含图片)

## ⚙️ 部署与使用

### 环境准备

1.  确保已安装 Python 3.8 或更高版本。
2.  安装项目依赖：
    ```bash
    pip install -r astrabot_plugin_self_learning/requirements.txt
    ```

### 配置说明

插件的配置通过 `astrabot_plugin_self_learning/config.py` 中的 `PluginConfig` 类进行管理，并可通过 AstrBot 的主配置进行覆盖。关键配置参数包括：

-   **基础开关**: `enable_message_capture` (消息抓取), `enable_auto_learning` (自动学习), `enable_realtime_learning` (实时学习), `enable_web_interface` (Web 管理界面)。
-   **目标设置**: `target_qq_list` (指定 QQ 号列表，空则抓取所有), `current_persona_name` (当前目标人格名称)。
-   **模型配置**:
    -   `filter_model_name`: 用于初步筛选的 LLM 模型（建议轻量级，如 `gpt-4o-mini`）。
    -   `refine_model_name`: 用于深度提炼和分析的 LLM 模型（建议高性能，如 `gpt-4o`）。
    -   `reinforce_model_name`: 用于强化学习的 LLM 模型。
    -   支持为每个模型配置独立的 `provider_id`, `api_url`, `api_key`，实现灵活的模型切换和私有化部署。
-   **学习参数**: `learning_interval_hours` (学习间隔), `min_messages_for_learning` (最少消息数), `max_messages_per_batch` (批处理大小)。
-   **筛选参数**: `message_min_length`, `message_max_length`, `confidence_threshold` (LLM 筛选置信度)。
-   **人格备份**: `auto_backup_enabled`, `backup_interval_hours`, `max_backups_per_group`。
-   **存储路径**: `data_dir`, `messages_db_path`, `learning_log_path` (默认自动生成)。

### 运行插件

本插件作为 AstrBot 框架的扩展，需通过 AstrBot 的插件管理机制加载和运行。具体加载和启用步骤请参考 AstrBot 框架的官方文档。

### 命令行工具

插件提供了一系列命令行接口，用于管理学习过程和数据：

-   `/learning_status`: 查看详细学习状态和统计信息。
-   `/start_learning`: 手动启动自动学习循环。
-   `/stop_learning`: 停止自动学习循环。
-   `/force_learning`: 强制执行一次完整的学习周期。
-   `/clear_data`: 清空所有学习数据（**请谨慎使用**）。
-   `/export_data`: 导出学习数据为 JSON 格式。

## 📊 数据管理与隐私

### 数据存储架构

-   **全局消息数据库 (`messages.db`)**: 统一存储所有原始消息 (`raw_messages`) 和经过 LLM 筛选及评分后的消息 (`filtered_messages`)，以及学习批次记录 (`learning_batches`)。
-   **分群数据库 (`group_databases/<group_id>_ID.db`)**: 为每个群聊或私聊会话独立存储用户画像 (`user_profiles`)、社交关系 (`social_relations`)、风格档案 (`style_profiles`) 和人格备份 (`persona_backups`)。这种设计确保了数据隔离和高效管理。
-   **表结构设计**: 各表均包含时间戳、ID 等字段，并针对查询效率创建了索引。`filtered_messages` 表新增 `quality_scores` 字段以 JSON 格式存储多维度评分。
-   **数据生命周期**: 支持自动清理过期数据、备份轮转和存储优化。

### 数据隐私与安全

-   所有数据默认本地存储，不上传至云端，确保用户数据隐私。
-   支持可选的数据加密存储（需额外配置或开发）。
-   用户对自身数据拥有完全控制权，可随时进行删除和导出。
-   遵循数据最小化原则，仅收集和处理必要数据。

## 🛡️ 质量保障与资源管理

### 学习质量保障

-   **多级质量监控**: `LearningQualityMonitor` 实时评估学习过程中的数据质量和模型输出，包括一致性、稳定性、风格偏移检测。
-   **回滚保护**: `PersonaBackupManager` 在每次人格更新前自动创建完整备份，支持一键恢复到历史版本，降低风险。
-   **人工介入点**: 关键节点（如质量得分过低）可触发警报，建议人工检查。

### 资源管理与优化

-   **异步架构**: 全异步消息处理管道和数据库操作，避免阻塞主线程，提高并发性能。
-   **内存优化**: `MessageCollectorService` 通过消息缓存和批量写入机制，以及 `MLAnalyzer` 中的样本大小控制，限制内存占用。
-   **CPU 节流**: 异步处理和智能批次调度，避免 CPU 过载。
-   **网络优化**: LLMClient 统一管理 API 调用，支持批量请求，减少网络开销。
-   **磁盘管理**: 自动清理过期数据，优化存储空间。

## 🔮 未来规划

-   开发基于 Web 的可视化管理界面，提供更直观的配置、监控和数据分析能力。
-   扩展社交关系分析维度，深入挖掘用户互动模式和群体动力学。
-   增强情感智能和语境理解能力，使机器人回复更具情商和上下文感知。
-   支持多语言对话风格学习和跨语言人格迁移。
-   探索分布式学习架构，提升处理大规模数据的能力和可伸缩性。
-   集成更多先进的机器学习和深度学习模型，提升分析和学习的精度。

## 📝 开发说明

本插件充分利用了现代异步编程范式和大型语言模型技术，构建了一个端到端的对话风格学习和人格优化系统。其模块化设计、可配置性、健壮的数据管理和质量保障机制，使其能够灵活适应不同的应用场景和性能需求，为 AstrBot 提供了强大的自主学习能力。
