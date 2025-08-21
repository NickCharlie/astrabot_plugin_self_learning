"""
LLM Prompt 静态文件
"""

# ml_analyzer.py 中的 prompt
ML_ANALYZER_REPLAY_MEMORY_SYSTEM_PROMPT = """
你是一个人格提炼专家。你的任务是分析以下消息记录，并结合当前人格描述，提炼出新的、更丰富的人格特征和对话风格。
重点关注消息中体现的：
- 语言习惯、用词偏好
- 情感表达方式
- 互动模式
- 知识领域和兴趣点
- 与当前人格的契合点和差异点

当前人格描述：
{current_persona_description}

请以结构化的JSON格式返回提炼结果，例如：
{{
    "new_style_features": {{
        "formal_level": 0.X,
        "enthusiasm_level": 0.Y,
        "question_tendency": 0.Z
    }},
    "new_topic_preferences": {{
        "话题A": 0.A,
        "话题B": 0.B
    }},
    "personality_insights": "一段关于人格演变的总结"
}}
"""

ML_ANALYZER_REPLAY_MEMORY_PROMPT = """
请分析以下消息记录，并结合当前人格，提炼出新的风格和特征：

{messages_text}
"""

ML_ANALYZER_SENTIMENT_ANALYSIS_PROMPT = """
请分析以下消息集合的整体情感倾向，并以JSON格式返回积极、消极、中性、疑问、惊讶五种情感的平均置信度分数（0-1之间）。

消息集合：
{messages_text}

请只返回一个JSON对象，例如：
{{
    "积极": 0.8,
    "消极": 0.1,
    "中性": 0.1,
    "疑问": 0.0,
    "惊讶": 0.0
}}
"""

# style_analyzer.py 中的 prompt
STYLE_ANALYZER_GENERATE_STYLE_ANALYSIS_PROMPT = """
请对以下对话文本进行详细的风格分析，以JSON格式返回结果：

对话文本：
{text}

请从以下维度进行分析并返回JSON格式结果：
{{
    "语言特色": {{
        "词汇使用": "分析词汇选择和使用特点",
        "句式结构": "分析句子结构和复杂度",
        "修辞手法": "识别使用的修辞技巧"
    }},
    "情感表达": {{
        "情感倾向": "整体情感倾向(积极/消极/中性)",
        "情感强度": "情感表达的强烈程度(0-1)",
        "情感变化": "情感在对话中的变化模式"
    }},
    "交流风格": {{
        "互动方式": "与他人交流的方式方式特点",
        "话题偏好": "倾向于讨论的话题类型",
        "回应模式": "对他人消息的回应特征"
    }},
    "个性化特征": {{
        "独特表达": "特有的表达习惯和用词",
        "思维模式": "体现的思维特点",
        "沟通目标": "沟通时的主要目标"
    }},
    "适应建议": {{
        "风格匹配度": "与目标人格的匹配程度(0-1)",
        "改进方向": "建议的风格调整方向",
        "学习价值": "作为学习材料的价值评估(0-1)"
    }}
}}
"""

STYLE_ANALYZER_EXTRACT_STYLE_PROFILE_PROMPT = """
请对以下对话文本进行数值化的风格特征提取，返回JSON格式的评分(0-1)：

对话文本：
{text}

请返回以下格式的JSON，每个维度给出0-1的评分：
{{
    "vocabulary_richness": 0.0,  // 词汇丰富度
    "sentence_complexity": 0.0,  // 句式复杂度
    "emotional_expression": 0.0,  // 情感表达度
    "interaction_tendency": 0.0,  // 互动倾向
    "topic_diversity": 0.0,       // 话题多样性
    "formality_level": 0.0,       // 正式程度
    "creativity_score": 0.0       // 创造性得分
}}
"""

STYLE_ANALYZER_GENERATE_STYLE_RECOMMENDATIONS_PROMPT = """
基于当前的风格档案数据和目标人格，生成风格优化建议：

当前风格档案：
{current_style_data}

目标人格：{target_persona}

请返回JSON格式的优化建议：
{{
    "优化方向": {{
        "需要加强": ["具体的风格维度和建议"],
        "需要调整": ["需要调整的方面"],
        "保持现状": ["已经较好的方面"]
    }},
    "具体建议": {{
        "词汇使用": "词汇选择的具体建议",
        "句式结构": "句式调整建议", 
        "情感表达": "情感表达优化建议",
        "互动方式": "互动方式改进建议"
    }},
    "实施策略": {{
        "短期目标": "1-2周内可以改进的方面",
        "中期目标": "1-2个月的改进方向",
        "长期目标": "长期的风格发展目标"
    }},
    "风险提示": "需要注意的潜在风险和副作用"
}}
"""

# multidimensional_analyzer.py 中的 prompt
MULTIDIMENSIONAL_ANALYZER_FILTER_MESSAGE_PROMPT = """
你是一个消息筛选专家，你的任务是判断一条消息是否具有以下特征：
1. 与当前人格的对话风格和兴趣高度匹配。
2. 消息内容特征鲜明，不平淡，具有一定的独特性或深度。
3. 对学习当前人格的对话模式和知识有积极意义。

当前人格描述：
{current_persona_description}

待筛选消息：
"{message_text}"

请你根据以上标准，对这条消息进行评估，并给出一个0到1之间的置信度分数。
0表示完全不符合，1表示完全符合。
请只返回一个0-1之间的数值，不需要其他说明。
"""

MULTIDIMENSIONAL_ANALYZER_EVALUATE_MESSAGE_QUALITY_PROMPT = """
你是一个专业的对话质量评估专家，请根据以下标准对一条消息进行多维度量化评分。
评分范围为0到1，0表示非常低，1表示非常高。

当前人格描述：
{current_persona_description}

待评估消息：
"{message_text}"

请评估以下维度并以JSON格式返回结果：
{{
    "content_quality": 0.0-1.0,  // 消息的深度、信息量、原创性、表达清晰度
    "relevance": 0.0-1.0,        // 与当前对话主题或人格的相关性
    "emotional_positivity": 0.0-1.0, // 消息的情感倾向（积极程度）
    "interactivity": 0.0-1.0,    // 消息是否引发或回应了互动（如提问、回应、@他人）
    "learning_value": 0.0-1.0    // 消息对模型学习当前人格对话模式和知识的潜在贡献
}}

请确保返回有效的JSON格式，并且只包含JSON对象，不需要其他说明。
"""

MULTIDIMENSIONAL_ANALYZER_EMOTIONAL_CONTEXT_PROMPT = """
请分析以下文本的情感倾向，并以JSON格式返回积极、消极、中性、疑问、惊讶五种情感的置信度分数（0-1之间）。

文本内容："{message_text}"

请只返回一个JSON对象，例如：
{{
    "积极": 0.8,
    "消极": 0.1,
    "中性": 0.1,
    "疑问": 0.0,
    "惊讶": 0.0
}}
"""

MULTIDIMENSIONAL_ANALYZER_FORMAL_LEVEL_PROMPT = """
请分析以下文本的正式程度，从0-1评分，0表示非常随意，1表示非常正式。

分析维度：
- 称谓使用（您/你）
- 语言风格（书面语/口语）
- 礼貌用语频率
- 句式结构复杂度
- 专业术语使用

文本内容："{text}"

请只返回一个0-1之间的数值，不需要其他说明。
"""

MULTIDIMENSIONAL_ANALYZER_ENTHUSIASM_LEVEL_PROMPT = """
请分析以下文本的热情程度，从0-1评分，0表示非常冷淡，1表示非常热情。

分析维度：
- 感叹号使用频率
- 积极情感词汇
- 表情符号使用
- 语气强烈程度
- 互动意愿表达

文本内容："{text}"

请只返回一个0-1之间的数值，不需要其他说明。
"""

MULTIDIMENSIONAL_ANALYZER_QUESTION_TENDENCY_PROMPT = """
请分析以下文本的提问倾向，从0-1评分，0表示完全没有疑问，1表示强烈的求知欲和疑问。

分析维度：
- 疑问句数量
- 求知欲表达
- 不确定性表述
- 征求意见的语气
- 探索性语言

文本内容："{text}"

请只返回一个0-1之间的数值，不需要其他说明。
"""

MULTIDIMENSIONAL_ANALYZER_DEEP_INSIGHTS_PROMPT = """
请基于以下用户数据，生成深度的用户画像洞察。以JSON格式返回结果：

用户数据：
{user_data_summary}

请分析以下维度并返回JSON格式结果：
{{
    "personality_type": "用户性格类型(如：外向型/内向型/混合型)",
    "communication_preference": "沟通偏好描述",
    "social_role": "在群体中的角色定位",
    "activity_pattern_analysis": "活动模式分析",
    "interest_alignment": "兴趣领域归类",
    "learning_potential": "学习价值评估(0-1)",
    "interaction_style": "互动风格特征",
    "content_contribution": "内容贡献度评估"
}}

请确保返回有效的JSON格式。
"""

MULTIDIMENSIONAL_ANALYZER_PERSONALITY_TRAITS_PROMPT = """
基于用户的沟通风格数据，分析其人格特质。请返回JSON格式的五大人格特质评分(0-1)：

沟通风格数据：
{communication_style_data}

请返回以下格式的JSON：
{{
    "openness": 0.0-1.0,  // 开放性
    "conscientiousness": 0.0-1.0,  // 尽责性  
    "extraversion": 0.0-1.0,  // 外向性
    "agreeableness": 0.0-1.0,  // 宜人性
    "neuroticism": 0.0-1.0  // 神经质
}}
"""

# factory.py 中 MessageFilter 的 prompt
MESSAGE_FILTER_SUITABLE_FOR_LEARNING_PROMPT = """
请判断以下消息是否与当前人格匹配，特征鲜明，且具有学习意义。
当前人格描述: {current_persona}
消息内容: "{message}"

请以 JSON 格式返回判断结果，包含 'suitable' (布尔值) 和 'confidence' (0.0-1.0 之间的浮点数)。
例如: {{"suitable": true, "confidence": 0.9}}
"""

# intelligent_responder.py 中的 prompt
INTELLIGENT_RESPONDER_DEFAULT_PERSONA_PROMPT = """

"""

# learning_quality_monitor.py 中缺失的 prompt
LEARNING_QUALITY_MONITOR_EMOTIONAL_BALANCE_PROMPT = """
请分析以下学习批次中消息的情感平衡性。评估消息集合在情感维度上是否多样化和平衡。

消息批次数据：
{batch_messages}

请从以下维度分析：
1. 情感多样性 - 包含多种情感表达（积极、消极、中性）
2. 情感强度分布 - 强烈情感与温和情感的平衡
3. 情感稳定性 - 情感表达是否合理稳定
4. 学习价值 - 这种情感平衡对人格学习是否有价值

请以JSON格式返回分析结果：
{{
    "emotional_diversity": 0.0-1.0,  // 情感多样性得分
    "intensity_balance": 0.0-1.0,    // 强度平衡得分
    "emotional_stability": 0.0-1.0,   // 情感稳定性得分
    "learning_value": 0.0-1.0,       // 学习价值得分
    "overall_balance": 0.0-1.0,      // 总体情感平衡得分
    "analysis_summary": "分析总结"
}}
"""

LEARNING_QUALITY_MONITOR_CONSISTENCY_PROMPT = """
请分析以下两个人格描述之间的一致性程度，评估人格更新前后的连贯性和兼容性。

原始人格描述：
{original_persona_prompt}

更新后人格描述：
{updated_persona_prompt}

请从以下维度评估一致性：
1. 核心价值观和性格特征是否保持
2. 语言风格和表达习惯是否延续
3. 兴趣爱好和知识领域是否兼容
4. 行为模式和互动方式是否协调
5. 整体人格形象是否和谐统一

请返回一个0-1之间的一致性得分，0表示完全不一致，1表示完全一致。
只返回数值，不需要其他解释。
"""

# progressive_learning.py 中的 prompt
PROGRESSIVE_LEARNING_GENERATE_UPDATED_PERSONA_PROMPT = """
基于当前人格和风格分析结果，生成更新后的人格描述。

当前人格信息：
{current_persona_json}

风格分析结果：
{style_analysis_json}

请根据风格分析结果对人格进行渐进式更新，确保：
1. 保持核心人格特征不变
2. 根据风格分析适当调整表达方式
3. 增强与分析结果匹配的特征
4. 保持整体人格的一致性和连贯性

请以JSON格式返回更新后的完整人格信息：
{{
    "name": "更新后的人格名称",
    "prompt": "更新后的完整人格描述",
    "begin_dialogs": ["开场对话列表"],
    "mood_imitation_dialogs": ["情绪模拟对话列表"]
}}
"""
