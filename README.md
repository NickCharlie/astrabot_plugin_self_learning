# AstrBot 智能自学习插件 🧠✨

## 🚀 项目概述

AstrBot 智能自学习插件是一个为 AstrBot 框架设计的**全功能 AI 自主学习解决方案**。该插件通过机器学习、多维度数据分析、情感智能系统和动态人格优化，为聊天Bot提供了**完整的自主学习生态系统**。

## 目前插件正在测试阶段 有许多Bug还没有修好 

## 欢迎加入QQ群聊 1021544792 反馈你所遇到的Bug

### 🌟 核心特性

- **🔄 全自动学习循环**: 实时消息捕获、智能筛选、风格分析、人格优化
- **🧠 情感智能系统**: 好感度管理、情绪状态、动态响应机制
- **📊 数据可视化分析**: 学习轨迹图表、用户行为分析、社交关系可视化
- **🤖 高级学习机制**: 人格切换、上下文感知学习、增量学习、对抗学习
- **💬 增强交互能力**: 多轮对话管理、跨群记忆、主动话题引导
- **🎯 智能化提升**: 知识图谱、个性化推荐、自适应学习率调整
- **🌐 Web 管理界面**: 完整的可视化管理控制台

## **<u>后台管理使用教程</u>**

### **<u>重要安全提醒</u>**

**<u>插件启动后请立即访问后台管理页面并修改默认密码！</u>**

### 🌐 访问后台管理

1. **启动插件后**，Web管理界面将在以下地址启动：
   ```
   http://localhost:7833 或 http://你的服务器IP:7833
   ```

2. **首次登录**：
   - 默认密码：`self_learning_pwd`
   - **<u>⚠️ 强烈建议：首次登录后立即修改密码！</u>**

### 🛡️ 安全说明

- **<u>请务必在生产环境中修改默认密码！</u>**

### 🎯 核心服务层 (`services/`)

#### 📊 数据分析与可视化服务
- **`data_analytics.py`**: 学习过程可视化、用户行为分析、社交网络图谱生成
- **功能**: 生成学习轨迹图表、用户活跃度热力图、话题趋势分析、社交关系可视化

#### 🧠 高级学习机制服务  
- **`advanced_learning.py`**: 人格切换、上下文感知学习、增量学习、对抗学习
- **功能**: 多场景人格自动切换、情境感知学习、知识增量更新、学习效果强化

#### 💬 增强交互服务
- **`enhanced_interaction.py`**: 多轮对话管理、跨群记忆、主动话题引导
- **功能**: 对话上下文跟踪、历史记忆管理、智能话题推荐、互动模式分析

#### 🎯 智能化提升服务
- **`intelligence_enhancement.py`**: 情感智能、知识图谱、个性化推荐、自适应学习
- **功能**: 情感状态识别、知识实体管理、智能推荐算法、学习率动态调整

#### ❤️ 好感度管理服务
- **`affection_manager.py`**: 用户好感度系统、bot情绪管理、动态情感响应
- **功能**: 
  - 用户好感度跟踪（单用户最大100分，总分250分上限）
  - 每日随机情绪系统（10种情绪类型）
  - 智能交互分析（称赞、鼓励、侮辱、骚扰等识别）
  - 动态情绪响应（根据用户行为自动调节bot情绪）
  - 好感度影响系统提示词（情绪状态融入AI回复）

#### 🔧 基础核心服务
- **`message_collector.py`**: 智能消息收集与预处理
- **`database_manager.py`**: 统一数据管理（全局+分群数据库架构）
- **`multidimensional_analyzer.py`**: 多维度消息分析与用户画像构建
- **`style_analyzer.py`**: 深度对话风格分析与量化
- **`learning_quality_monitor.py`**: 学习质量实时监控与评估
- **`progressive_learning.py`**: 渐进式学习流程协调
- **`ml_analyzer.py`**: 机器学习增强分析
- **`persona_manager.py`**: 动态人格管理
- **`persona_updater.py`**: 智能人格更新
- **`persona_backup_manager.py`**: 人格数据备份与恢复

### 🎮 用户命令接口

#### 基础命令
- `/learning_status` - 查看详细学习状态和统计信息
- `/start_learning` - 手动启动自动学习循环  
- `/stop_learning` - 停止自动学习循环
- `/force_learning` - 强制执行一次完整学习周期
- `/clear_data` - 清空学习数据（**请谨慎使用**）
- `/export_data` - 导出学习数据为JSON格式

#### 新增高级命令
- `/affection_status` - 查看好感度系统状态（好感度排行榜、当前情绪等）
- `/set_mood <情绪类型>` - 手动设置bot情绪状态（如happy、sad、excited等）
- `/analytics_report` - 生成数据分析报告（学习统计、用户行为模式等）
- `/persona_switch <人格名称>` - 切换到指定人格模式

### 🔄 智能运行逻辑

#### 1. **消息处理流程**
```
用户消息 → QQ过滤 → 消息收集 → 好感度处理 → 增强交互更新 → 实时学习处理
```

#### 2. **好感度系统流程**
```
消息分析 → 交互类型识别 → 好感度计算 → 情绪状态更新 → 系统提示词调整
```

#### 3. **学习循环流程**  
```
消息筛选 → 多维度分析 → 风格提取 → 质量评估 → 人格更新 → 效果验证
```

#### 4. **情感智能流程**
```
情感识别 → 知识图谱更新 → 个性化推荐 → 自适应调整 → 响应生成
```

## 🛠️ 技术栈升级

### 🔥 AI/ML 技术栈
- **大型语言模型**: OpenAI GPT系列、自定义API支持
- **机器学习**: `scikit-learn`、`numpy`、`pandas`
- **情感计算**: 情绪识别、情感状态建模
- **知识图谱**: `networkx`、关系网络分析
- **自然语言处理**: `jieba`、`nltk`、`spacy`

### 📊 数据可视化
- **图表生成**: `plotly`、`matplotlib`、`seaborn`
- **网络可视化**: `bokeh`
- **数据分析**: 多维度统计分析

### 🏗️ 系统架构
- **异步框架**: `asyncio`、`aiohttp`、`aiofiles`
- **数据库**: `aiosqlite`、分布式数据存储
- **Web框架**: `quart`、`quart-cors`
- **缓存系统**: `cachetools`、`redis`

## 📋 详细配置参数解析

本插件提供了丰富的配置选项，支持高度自定义的学习和交互行为。

### 🔧 基础学习设置 (Self_Learning_Basic)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_message_capture` | bool | true | 是否启用消息抓取功能，关闭后插件停止收集新消息 |
| `enable_auto_learning` | bool | true | 是否启用定时自动学习，关闭后需要手动触发学习 |
| `enable_realtime_learning` | bool | false | 是否在收到消息时立即处理，会增加实时负载 |
| `enable_web_interface` | bool | true | 是否启用Web管理界面用于查看和管理学习数据 |

### 🎯 目标设置 (Target_Settings)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_qq_list` | list | [] | 指定要学习的QQ号列表，为空则学习所有用户消息 |
| `current_persona_name` | string | "default" | 插件将学习并优化此人格的对话风格 |

### 🤖 模型配置 (Model_Configuration)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `filter_model_name` | string | "gpt-4o-mini" | 用于初步筛选消息的弱模型，建议使用速度快、成本低的模型 |
| `refine_model_name` | string | "gpt-4o" | 用于深度分析和提炼对话风格的强模型 |
| `reinforce_model_name` | string | "gpt-4o" | 用于强化学习的LLM模型 |
| `filter_provider_id` | string | null | 筛选模型的LLM提供商ID，为空使用默认提供商 |
| `refine_provider_id` | string | null | 提炼模型的LLM提供商ID，为空使用默认提供商 |
| `reinforce_provider_id` | string | null | 强化模型的LLM提供商ID，为空使用默认提供商 |
| `filter_api_url` | string | null | 自定义筛选模型的API接口地址 |
| `filter_api_key` | string | null | 自定义筛选模型的API密钥 |
| `refine_api_url` | string | null | 自定义提炼模型的API接口地址 |
| `refine_api_key` | string | null | 自定义提炼模型的API密钥 |
| `reinforce_api_url` | string | null | 自定义强化模型的API接口地址 |
| `reinforce_api_key` | string | null | 自定义强化模型的API密钥 |

### ⏰ 学习参数 (Learning_Parameters)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `learning_interval_hours` | int | 6 | 自动学习的时间间隔，单位为小时 |
| `min_messages_for_learning` | int | 50 | 开始学习所需的最少消息数量 |
| `max_messages_per_batch` | int | 200 | 单次学习处理的最大消息数量，避免一次处理过多消息 |

### 🔍 筛选参数 (Filter_Parameters)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `message_min_length` | int | 5 | 参与学习的消息最小字符长度 |
| `message_max_length` | int | 500 | 参与学习的消息最大字符长度 |
| `confidence_threshold` | float | 0.7 | 消息筛选的置信度阈值，0-1之间，越高越严格 |

### 🎨 风格分析 (Style_Analysis)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `style_analysis_batch_size` | int | 100 | 单次风格分析处理的消息数量 |
| `style_update_threshold` | float | 0.8 | 触发人格风格更新的置信度阈值，0-1之间 |

### 🔬 机器学习设置 (Machine_Learning_Settings)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_ml_analysis` | bool | true | 是否启用scikit-learn进行文本聚类和行为分析 |
| `max_ml_sample_size` | int | 100 | 机器学习分析的最大样本数量，控制资源使用 |
| `ml_cache_timeout_hours` | int | 1 | 机器学习分析结果的缓存时间 |


### 💾 人格备份设置 (Persona_Backup_Settings)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auto_backup_enabled` | bool | true | 是否在人格更新前自动创建备份 |
| `backup_interval_hours` | int | 24 | 自动备份的时间间隔 |
| `max_backups_per_group` | int | 10 | 每个群保留的最大备份数量 |

### ❤️ 好感度系统设置 (Affection_System_Settings)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_affection_system` | bool | true | 是否启用用户好感度和情绪响应系统 |
| `max_total_affection` | int | 250 | bot对所有用户的总好感度上限值 |
| `max_user_affection` | int | 100 | 单个用户可获得的最大好感度 |
| `affection_decay_rate` | float | 0.95 | 好感度重新分配时的衰减比例，0-1之间 |
| `daily_mood_change` | bool | true | 是否每天随机更换bot的情绪状态 |
| `mood_affect_affection` | bool | true | 当前情绪是否影响好感度变化幅度 |

### 🎭 情绪系统设置 (Mood_System_Settings)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_daily_mood` | bool | true | 是否启用每日随机情绪系统 |
| `mood_change_hour` | int | 6 | 每日更新情绪的小时(0-23) |
| `mood_persistence_hours` | int | 24 | 每次情绪状态持续的小时数 |

### ⚙️ 高级设置 (Advanced_Settings)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debug_mode` | bool | false | 启用详细的调试日志输出 |
| `save_raw_messages` | bool | true | 是否保存未经处理的原始消息用于分析 |
| `auto_backup_interval_days` | int | 7 | 学习数据自动备份的间隔天数，0为禁用 |

### 💡 配置建议

1. **生产环境建议**: 
   - 关闭 `debug_mode` 以提高性能
   - 适当调整 `learning_interval_hours` 避免过于频繁的学习
   - 根据服务器性能调整 `max_messages_per_batch`

2. **开发测试建议**:
   - 启用 `debug_mode` 便于调试
   - 降低 `min_messages_for_learning` 快速测试学习功能
   - 启用 `enable_realtime_learning` 实时查看效果

3. **资源优化建议**:
   - 合理设置 `max_ml_sample_size` 控制内存使用
   - 调整 `ml_cache_timeout_hours` 平衡性能与实时性
   - 定期清理过期备份，控制存储空间

### 新增配置项

#### 好感度系统配置
```python
enable_affection_system: bool = True      # 启用好感度系统
max_total_affection: int = 250           # bot总好感度上限
max_user_affection: int = 100            # 单用户好感度上限  
affection_decay_rate: float = 0.95       # 好感度衰减比例
daily_mood_change: bool = True           # 启用每日情绪变化
mood_affect_affection: bool = True       # 情绪影响好感度变化
```

#### 情绪系统配置
```python
enable_daily_mood: bool = True           # 启用每日情绪
mood_change_hour: int = 6                # 情绪更新时间（24小时制）  
mood_persistence_hours: int = 24         # 情绪持续时间
```

#### Web界面配置
```python
enable_web_interface: bool = True        # 启用Web管理界面
web_interface_port: int = 7833          # Web界面端口
```

## 💾 数据管理架构升级

### 🗄️ 数据库设计

#### 新增数据表
- **`user_affection`**: 用户好感度记录
- **`bot_mood`**: bot情绪状态历史  
- **`affection_history`**: 好感度变化记录
- **`emotion_profiles`**: 用户情感档案
- **`knowledge_entities`**: 知识实体库
- **`user_preferences`**: 用户偏好设置
- **`conversation_contexts`**: 对话上下文管理

### 🔐 数据隐私与安全
- **本地存储**: 所有数据本地化，确保隐私安全
- **数据加密**: 敏感信息加密存储
- **访问控制**: Web界面密码保护
- **数据备份**: 自动备份与恢复机制

## 🚀 部署与使用

### 环境准备
1. 确保已安装 Python 3.8+ 
2. 安装项目依赖：
   ```bash
   pip install -r astrabot_plugin_self_learning/requirements.txt
   ```

### 快速开始
1. 将插件添加到AstrBot插件目录
2. 启动AstrBot，插件将自动加载
3. 访问Web管理界面：`http://localhost:7833`
4. 使用默认密码登录并立即修改密码
5. 在Astrbot后台插件管理中设置插件配置项

## 🎯 智能特性展示

### ❤️ 情感智能系统
- **动态好感度**: 根据用户互动自动调节好感度
- **情绪识别**: 智能识别夸赞、鼓励、侮辱、骚扰等交互类型
- **情绪响应**: bot情绪会根据用户行为动态变化
- **情感融入**: 当前情绪状态影响AI回复的语调和内容

### 📊 数据可视化分析
- **学习轨迹图**: 可视化学习进度和质量变化
- **用户行为热力图**: 分析用户活跃模式
- **社交网络图**: 展示群内用户关系网络
- **情感趋势分析**: 跟踪群聊情感氛围变化

### 🧠 智能学习机制
- **场景感知**: 根据不同场景自动切换最适合的人格
- **增量学习**: 持续学习新知识，不遗忘历史经验  
- **质量监控**: 实时评估学习效果，自动调优
- **个性化推荐**: 基于用户偏好推荐话题和回复策略

## 🤝 贡献指南

欢迎开发者参与项目建设！
- **Bug反馈**: 使用GitHub Issues报告问题
- **功能建议**: 提交Feature Request  
- **代码贡献**: Fork项目并提交Pull Request
- **文档改进**: 帮助完善文档和教程
