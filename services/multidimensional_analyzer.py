"""
多维度学习引擎 - 全方位分析用户特征和社交关系
"""
import re
import json
import time
import asyncio # 确保 asyncio 导入
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import emoji # 导入 emoji 库

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..config import PluginConfig
from ..exceptions import StyleAnalysisError
from ..core.llm_client import LLMClient # 导入自定义LLMClient
from .database_manager import DatabaseManager # 导入 DatabaseManager


@dataclass
class UserProfile:
    """用户画像"""
    qq_id: str
    qq_name: str
    nicknames: List[str] = None
    activity_pattern: Dict[str, Any] = None
    communication_style: Dict[str, float] = None
    social_connections: List[str] = None
    topic_preferences: Dict[str, float] = None
    emotional_tendency: Dict[str, float] = None
    
    def __post_init__(self):
        if self.nicknames is None:
            self.nicknames = []
        if self.activity_pattern is None:
            self.activity_pattern = {}
        if self.communication_style is None:
            self.communication_style = {}
        if self.social_connections is None:
            self.social_connections = []
        if self.topic_preferences is None:
            self.topic_preferences = {}
        if self.emotional_tendency is None:
            self.emotional_tendency = {}


@dataclass
class SocialRelation:
    """社交关系"""
    from_user: str
    to_user: str
    relation_type: str  # mention, reply, frequent_interaction
    strength: float  # 关系强度 0-1
    frequency: int   # 交互频次
    last_interaction: str


@dataclass
class ContextualPattern:
    """情境模式"""
    context_type: str  # time_based, topic_based, social_based
    pattern_name: str
    triggers: List[str]
    characteristics: Dict[str, Any]
    confidence: float


class MultidimensionalAnalyzer:
    """多维度分析器"""
    
    def __init__(self, config: PluginConfig, db_manager: DatabaseManager, context=None,
                 filter_llm_client: Optional[LLMClient] = None,
                 refine_llm_client: Optional[LLMClient] = None,
                 reinforce_llm_client: Optional[LLMClient] = None,
                 prompts: Any = None): # 添加 prompts 参数
        self.config = config
        self.context = context
        self.db_manager: DatabaseManager = db_manager # 直接传入 DatabaseManager 实例
        
        # LLM 客户端通过参数传入
        self.filter_llm_client = filter_llm_client
        self.refine_llm_client = refine_llm_client
        self.reinforce_llm_client = reinforce_llm_client
        self.prompts = prompts # 保存 prompts

        if not (self.filter_llm_client and config.filter_api_url and config.filter_api_key and config.filter_model_name):
            logger.warning(f"筛选模型LLM配置不完整：API URL: {config.filter_api_url}, API Key: {'***' if config.filter_api_key else 'None'}, Model Name: {config.filter_model_name}。将无法使用LLM进行消息筛选。")

        if not (self.refine_llm_client and config.refine_api_url and config.refine_api_key and config.refine_model_name):
            logger.warning(f"提炼模型LLM配置不完整：API URL: {config.refine_api_url}, API Key: {'***' if config.refine_api_key else 'None'}, Model Name: {config.refine_model_name}。将无法使用LLM进行深度分析。")

        if not (self.reinforce_llm_client and config.reinforce_api_url and config.reinforce_api_key and config.reinforce_model_name):
            logger.warning(f"强化模型LLM配置不完整：API URL: {config.reinforce_api_url}, API Key: {'***' if config.reinforce_api_key else 'None'}, Model Name: {config.reinforce_model_name}。将无法使用LLM进行强化学习。")
        
        # 用户画像存储
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # 社交关系图谱
        self.social_graph: Dict[str, List[SocialRelation]] = defaultdict(list)
        
        # 昵称映射表
        self.nickname_mapping: Dict[str, str] = {}  # nickname -> qq_id
        
        # 情境模式库
        self.contextual_patterns: List[ContextualPattern] = []
        
        # 话题分类器
        self.topic_keywords = {
            '日常聊天': ['吃饭', '睡觉', '上班', '下班', '休息', '忙'],
            '游戏娱乐': ['游戏', '电影', '音乐', '小说', '动漫', '综艺'],
            '学习工作': ['学习', '工作', '项目', '考试', '会议', '任务'],
            '情感交流': ['开心', '难过', '生气', '担心', '兴奋', '无聊'],
            '技术讨论': ['代码', '程序', '算法', '技术', '开发', '编程'],
            '生活分享': ['旅游', '美食', '购物', '健身', '宠物', '家庭']
        }
        
        logger.info("多维度学习引擎初始化完成")

    async def start(self):
        """服务启动时加载用户画像和社交关系"""
        # 假设每个群组有独立的画像和关系，这里需要一个 group_id
        # 为了简化，暂时假设加载一个默认的或全局的画像和关系
        # 实际应用中，可能需要根据当前处理的群组ID来加载
        default_group_id = "global_profiles" # 或者从配置中获取
        
        # 加载所有用户画像 (这里需要 DatabaseManager 提供 load_all_user_profiles 方法)
        # 暂时只加载一个示例用户
        # loaded_profile_data = await self.db_manager.load_user_profile(default_group_id, "example_qq_id")
        # if loaded_profile_data:
        #     self.user_profiles[loaded_profile_data['qq_id']] = UserProfile(**loaded_profile_data)
        #     logger.info(f"已从数据库加载用户画像: {loaded_profile_data['qq_id']}")
        
        # 加载所有社交关系 (这里需要 DatabaseManager 提供 load_all_social_relations 方法)
        # loaded_social_graph = await self.db_manager.load_social_graph(default_group_id)
        # for relation_data in loaded_social_graph:
        #     relation = SocialRelation(**relation_data)
        #     self.social_graph[relation.from_user].append(relation)
        # logger.info(f"已从数据库加载 {len(loaded_social_graph)} 条社交关系。")
        
        logger.info("多维度学习引擎启动，准备进行分析。")

    async def filter_message_with_llm(self, message_text: str, current_persona_description: str) -> bool:
        """
        使用 LLM 对消息进行智能筛选，判断其是否与当前人格匹配、特征鲜明且有学习意义。
        返回 True 表示消息通过筛选，False 表示不通过。
        """
        if not (self.config.filter_api_url and self.config.filter_api_key and self.config.filter_model_name):
            logger.warning("筛选模型LLM配置不完整，跳过LLM消息筛选。")
            return True

        prompt = self.prompts.MULTIDIMENSIONAL_ANALYZER_FILTER_MESSAGE_PROMPT.format(
            current_persona_description=current_persona_description,
            message_text=message_text
        )
        try:
            response = await self.filter_llm_client.chat_completion(
                prompt=prompt,
                api_url=self.config.filter_api_url,
                api_key=self.config.filter_api_key,
                model_name=self.config.filter_model_name
            )
            if response and response.text():
                numbers = re.findall(r'0\.\d+|1\.0|0', response.text().strip())
                if numbers:
                    confidence = min(float(numbers[0]), 1.0)
                    logger.debug(f"消息筛选置信度: {confidence} (阈值: {self.config.confidence_threshold})")
                    return confidence >= self.config.confidence_threshold
            logger.warning(f"LLM筛选模型未返回有效置信度，消息默认不通过筛选。")
            return False
        except Exception as e:
            logger.error(f"LLM消息筛选失败: {e}")
            return False

    async def evaluate_message_quality_with_llm(self, message_text: str, current_persona_description: str) -> Dict[str, float]:
        """
        使用 LLM 对消息进行多维度量化评分。
        评分维度包括：内容质量、相关性、情感积极性、互动性、学习价值。
        返回一个包含各维度评分的字典。
        """
        if not (self.config.refine_api_url and self.config.refine_api_key and self.config.refine_model_name):
            logger.warning("提炼模型LLM客户端未初始化，无法使用LLM进行多维度量化评分。")
            return {
                "content_quality": 0.5,
                "relevance": 0.5,
                "emotional_positivity": 0.5,
                "interactivity": 0.5,
                "learning_value": 0.5
            }

        prompt = self.prompts.MULTIDIMENSIONAL_ANALYZER_EVALUATE_MESSAGE_QUALITY_PROMPT.format(
            current_persona_description=current_persona_description,
            message_text=message_text
        )
        try:
            response = await self.refine_llm_client.chat_completion(
                prompt=prompt,
                api_url=self.config.refine_api_url,
                api_key=self.config.refine_api_key,
                model_name=self.config.refine_model_name
            )
            if response and response.text():
                try:
                    scores = json.loads(response.text().strip())
                    # 确保所有分数都在0-1之间
                    for key, value in scores.items():
                        scores[key] = max(0.0, min(float(value), 1.0))
                    logger.debug(f"消息多维度评分: {scores}")
                    return scores
                except json.JSONDecodeError:
                    logger.warning(f"LLM多维度评分响应JSON解析失败，返回默认评分。响应内容: {response.text()}")
                    return {
                        "content_quality": 0.5,
                        "relevance": 0.5,
                        "emotional_positivity": 0.5,
                        "interactivity": 0.5,
                        "learning_value": 0.5
                    }
            logger.warning(f"LLM多维度评分模型未返回有效响应，返回默认评分。")
            return {
                "content_quality": 0.5,
                "relevance": 0.5,
                "emotional_positivity": 0.5,
                "interactivity": 0.5,
                "learning_value": 0.5
            }
        except Exception as e:
            logger.error(f"LLM多维度评分失败: {e}")
            return {
                "content_quality": 0.5,
                "relevance": 0.5,
                "emotional_positivity": 0.5,
                "interactivity": 0.5,
                "learning_value": 0.5
            }

    async def analyze_message_context(self, event: AstrMessageEvent, message_text: str) -> Dict[str, Any]:
        """分析消息的多维度上下文"""
        try:
            sender_id = event.get_sender_id()
            sender_name = event.get_sender_name()
            group_id = event.get_group_id()
            
            # 更新用户画像
            await self._update_user_profile(group_id, sender_id, sender_name, message_text, event) # 传入 group_id
            
            # 分析社交关系
            social_context = await self._analyze_social_context(event, message_text)
            
            # 分析话题偏好
            topic_context = await self._analyze_topic_context(message_text)
            
            # 分析情感倾向
            emotional_context = await self._analyze_emotional_context(message_text)
            
            # 分析时间模式
            temporal_context = await self._analyze_temporal_context(event)
            
            # 分析沟通风格
            style_context = await self._analyze_communication_style(message_text)
            
            return {
                'user_profile': self.user_profiles.get(sender_id, {}).activity_pattern if sender_id in self.user_profiles else {},
                'social_context': social_context,
                'topic_context': topic_context,
                'emotional_context': emotional_context,
                'temporal_context': temporal_context,
                'style_context': style_context,
                'contextual_relevance': await self._calculate_contextual_relevance(
                    sender_id, message_text, event
                )
            }
            
        except Exception as e:
            logger.error(f"多维度上下文分析失败: {e}")
            return {}

    async def _update_user_profile(self, group_id: str, qq_id: str, qq_name: str, message_text: str, event: AstrMessageEvent):
        """更新用户画像并持久化"""
        profile_data = await self.db_manager.load_user_profile(group_id, qq_id)
        if profile_data:
            profile = UserProfile(**profile_data)
        else:
            profile = UserProfile(qq_id=qq_id, qq_name=qq_name)
        
        # 更新活动模式
        current_hour = datetime.now().hour
        if 'activity_hours' not in profile.activity_pattern:
            profile.activity_pattern['activity_hours'] = Counter()
        profile.activity_pattern['activity_hours'][current_hour] += 1
        
        # 更新消息长度偏好
        msg_length = len(message_text)
        if 'message_lengths' not in profile.activity_pattern:
            profile.activity_pattern['message_lengths'] = []
        profile.activity_pattern['message_lengths'].append(msg_length)
        
        # 保持最近100条消息的长度记录
        if len(profile.activity_pattern['message_lengths']) > 100:
            profile.activity_pattern['message_lengths'] = profile.activity_pattern['message_lengths'][-100:]
        
        # 更新话题偏好
        topics = await self._extract_topics(message_text)
        for topic in topics:
            if topic not in profile.topic_preferences:
                profile.topic_preferences[topic] = 0
            profile.topic_preferences[topic] += 1
        
        # 更新沟通风格
        style_features = await self._extract_style_features(message_text)
        for feature, value in style_features.items():
            if feature not in profile.communication_style:
                profile.communication_style[feature] = []
            profile.communication_style[feature].append(value)
            
            # 保持最近50个特征值
            if len(profile.communication_style[feature]) > 50:
                profile.communication_style[feature] = profile.communication_style[feature][-50:]
        
        self.user_profiles[qq_id] = profile # 更新内存中的画像
        await self.db_manager.save_user_profile(group_id, asdict(profile)) # 持久化到数据库

    async def _analyze_social_context(self, event: AstrMessageEvent, message_text: str) -> Dict[str, Any]:
        """分析社交关系上下文"""
        try:
            sender_id = event.get_sender_id()
            group_id = event.get_group_id()
            
            social_context = {
                'mentions': [],
                'replies': [],
                'interaction_strength': {},
                'group_role': 'member'
            }
            
            # 提取@消息
            mentions = self._extract_mentions(message_text)
            social_context['mentions'] = mentions
            
            # 更新社交关系
            for mentioned_user in mentions:
                await self._update_social_relation(
                    sender_id, mentioned_user, 'mention', group_id
                )
            
            # 分析回复关系（如果框架支持）
            if hasattr(event, 'get_reply_info') and event.get_reply_info():
                reply_info = event.get_reply_info()
                replied_user = reply_info.get('user_id')
                if replied_user:
                    social_context['replies'].append(replied_user)
                    await self._update_social_relation(
                        sender_id, replied_user, 'reply', group_id
                    )
            
            # 计算与群内成员的交互强度
            if sender_id in self.social_graph:
                for relation in self.social_graph[sender_id]:
                    social_context['interaction_strength'][relation.to_user] = relation.strength
            
            # 分析群内角色（基于发言频率和@次数）
            group_role = await self._analyze_group_role(sender_id, group_id)
            social_context['group_role'] = group_role
            
            return social_context
        
        except Exception as e:
            logger.warning(f"社交上下文分析失败: {e}")
            return {}

    async def _analyze_topic_context(self, message_text: str) -> Dict[str, float]:
        """分析话题上下文"""
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in message_text:
                    score += 1
            
            if score > 0:
                topic_scores[topic] = score / len(keywords)
        
        return topic_scores

    async def _analyze_emotional_context(self, message_text: str) -> Dict[str, float]:
        """使用LLM分析情感上下文"""
        if not (self.config.refine_api_url and self.config.refine_api_key and self.config.refine_model_name):
            logger.warning("提炼模型LLM客户端未初始化，无法使用LLM分析情感上下文，使用简化算法。")
            return self._simple_emotional_analysis(message_text)

        try:
            prompt = self.prompts.MULTIDIMENSIONAL_ANALYZER_EMOTIONAL_CONTEXT_PROMPT.format(
                message_text=message_text
            )
            try:
                response = await self.refine_llm_client.chat_completion(
                    prompt=prompt,
                    api_url=self.config.refine_api_url,
                    api_key=self.config.refine_api_key,
                    model_name=self.config.refine_model_name
                )
                
                if response and response.text():
                    try:
                        emotion_scores = json.loads(response.text().strip())
                        # 确保所有分数都在0-1之间
                        for key, value in emotion_scores.items():
                            emotion_scores[key] = max(0.0, min(float(value), 1.0))
                        return emotion_scores
                    except json.JSONDecodeError:
                        logger.warning(f"LLM响应JSON解析失败，返回简化情感分析。响应内容: {response.text()}")
                        return self._simple_emotional_analysis(message_text)
                return self._simple_emotional_analysis(message_text)
                
            except Exception as e:
                logger.warning(f"LLM情感分析失败，使用简化算法: {e}")
        except Exception as e:
            logger.warning(f"LLM情感分析失败 - 2，使用简化算法: {e}")


    def _simple_emotional_analysis(self, message_text: str) -> Dict[str, float]:
        """简化的情感分析（备用）"""
        emotions = {
            '积极': ['开心', '高兴', '兴奋', '满意', '喜欢', '爱', '好棒', '太好了', '哈哈', '😄', '😊', '👍'],
            '消极': ['难过', '生气', '失望', '无聊', '烦', '讨厌', '糟糕', '不好', '😭', '😢', '😡'],
            '中性': ['知道', '明白', '可以', '好的', '嗯', '哦', '这样', '然后'],
            '疑问': ['吗', '呢', '？', '什么', '怎么', '为什么', '哪里', '🤔'],
            '惊讶': ['哇', '天哪', '真的', '不会吧', '太', '竟然', '居然', '😱', '😯']
        }
        
        emotion_scores = {}
        # 将消息文本按空格或标点符号分割成单词，并过滤掉空字符串
        words = [word for word in re.split(r'\s+|[，。！？；：]', message_text) if word]
        total_words = len(words) # 已经修改为单词总数
        
        for emotion, keywords in emotions.items():
            count = 0
            for keyword in keywords:
                # 检查关键词是否在单词列表中
                count += words.count(keyword)
            
            emotion_scores[emotion] = count / max(total_words, 1)
        
        return emotion_scores

    async def _analyze_temporal_context(self, event: AstrMessageEvent) -> Dict[str, Any]:
        """分析时间上下文"""
        now = datetime.now()
        
        time_context = {
            'hour': now.hour,
            'weekday': now.weekday(),
            'time_period': self._get_time_period(now.hour),
            'is_weekend': now.weekday() >= 5,
            'season': self._get_season(now.month)
        }
        
        return time_context

    async def _analyze_communication_style(self, message_text: str) -> Dict[str, float]:
        """分析沟通风格"""
        style_features = {
            'formal_level': self._calculate_formal_level(message_text),
            'enthusiasm_level': self._calculate_enthusiasm_level(message_text),
            'question_tendency': self._calculate_question_tendency(message_text),
            'emoji_usage': self._calculate_emoji_usage(message_text),
            'length_preference': len(message_text),
            'punctuation_style': self._calculate_punctuation_style(message_text)
        }
        
        return style_features

    async def _extract_topics(self, message_text: str) -> List[str]:
        """提取消息话题"""
        detected_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in message_text:
                    detected_topics.append(topic)
                    break
        
        return detected_topics

    async def _extract_style_features(self, message_text: str) -> Dict[str, float]:
        """提取风格特征"""
        return {
            'length': len(message_text),
            'punctuation_ratio': len([c for c in message_text if c in '，。！？；：']) / max(len(message_text), 1),
            'emoji_count': emoji.emoji_count(message_text),
            'question_count': message_text.count('？') + message_text.count('?'),
            'exclamation_count': message_text.count('！') + message_text.count('!')
        }

    def _extract_mentions(self, message_text: str) -> List[str]:
        """提取@消息"""
        # 匹配@用户模式
        at_pattern = r'@(\w+|\d+)'
        matches = re.findall(at_pattern, message_text)
        
        # 尝试解析昵称到QQ号的映射
        mentioned_users = []
        for match in matches:
            if match.isdigit():
                # 直接@的QQ号
                mentioned_users.append(match)
            else:
                # @的昵称，尝试找到对应的QQ号
                if match in self.nickname_mapping:
                    mentioned_users.append(self.nickname_mapping[match])
                else:
                    # 记录未知昵称
                    mentioned_users.append(f"nickname:{match}")
        
        return mentioned_users

    async def _update_social_relation(self, from_user: str, to_user: str, relation_type: str, group_id: str):
        """更新社交关系"""
        # 查找现有关系
        existing_relation = None
        for relation in self.social_graph[from_user]:
            if relation.to_user == to_user and relation.relation_type == relation_type:
                existing_relation = relation
                break
        
        if existing_relation:
            # 更新现有关系
            existing_relation.frequency += 1
            existing_relation.last_interaction = datetime.now().isoformat()
            existing_relation.strength = min(existing_relation.strength + 0.1, 1.0)
        else:
            # 创建新关系
            new_relation = SocialRelation(
                from_user=from_user,
                to_user=to_user,
                relation_type=relation_type,
                strength=0.1,
                frequency=1,
                last_interaction=datetime.now().isoformat()
            )
            self.social_graph[from_user].append(new_relation)
        
        # 持久化社交关系
        await self.db_manager.save_social_relation(group_id, asdict(existing_relation if existing_relation else new_relation))

    async def _analyze_group_role(self, user_id: str, group_id: str) -> str:
        """分析用户在群内的角色"""
        # 这里可以基于发言频率、被@次数等判断用户角色
        # 简化实现
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            mention_count = sum(1 for relations in self.social_graph.values() 
                              for relation in relations 
                              if relation.to_user == user_id and relation.relation_type == 'mention')
            
            if mention_count > 10:
                return 'active_member'
            elif mention_count > 5:
                return 'regular_member'
            else:
                return 'member'
        
        return 'member'

    async def _calculate_contextual_relevance(self, sender_id: str, message_text: str, event: AstrMessageEvent) -> float:
        """计算上下文相关性得分"""
        relevance_score = 0.0
        
        # 基于用户历史行为的相关性
        if sender_id in self.user_profiles:
            profile = self.user_profiles[sender_id]
            
            # 话题一致性
            current_topics = await self._extract_topics(message_text)
            for topic in current_topics:
                if topic in profile.topic_preferences:
                    relevance_score += 0.2
            
            # 风格一致性
            current_style = await self._extract_style_features(message_text)
            if 'length' in profile.communication_style:
                avg_length = sum(profile.communication_style['length'][-10:]) / min(10, len(profile.communication_style['length']))
                length_similarity = 1.0 - abs(current_style['length'] - avg_length) / max(avg_length, 1)
                relevance_score += length_similarity * 0.1
        
            # 时间上下文相关性
            current_hour = datetime.now().hour
            if sender_id in self.user_profiles:
                profile = self.user_profiles[sender_id]
                if 'activity_hours' in profile.activity_pattern:
                    hour_frequency = profile.activity_pattern['activity_hours'].get(current_hour, 0)
                    total_messages = sum(profile.activity_pattern['activity_hours'].values())
                    if total_messages > 0:
                        time_relevance = hour_frequency / total_messages
                        relevance_score += time_relevance * 0.2
        
        return min(relevance_score, 1.0)

    def _get_time_period(self, hour: int) -> str:
        """获取时间段"""
        if 6 <= hour < 12:
            return '上午'
        elif 12 <= hour < 18:
            return '下午'
        elif 18 <= hour < 22:
            return '晚上'
        else:
            return '深夜'

    def _get_season(self, month: int) -> str:
        """获取季节"""
        if month in [1, 2, 12]:
            return '冬季'
        elif month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        else:
            return '秋季'

    async def _call_llm_for_style_analysis(self, text: str, prompt_template: str, fallback_function: callable, analysis_name: str) -> float:
        """
        通用的LLM风格分析辅助函数。
        Args:
            text: 待分析的文本。
            prompt_template: LLM提示模板。
            fallback_function: LLM客户端未初始化或调用失败时使用的备用函数。
            analysis_name: 分析名称，用于日志记录。
        Returns:
            0-1之间的评分。
        """
        if not (self.config.refine_api_url and self.config.refine_api_key and self.config.refine_model_name):
            logger.warning(f"提炼模型LLM客户端未初始化，无法使用LLM计算{analysis_name}，使用简化算法。")
            return fallback_function(text)

        try:
            prompt = prompt_template.format(text=text)
            response = await self.refine_llm_client.chat_completion(
                prompt=prompt,
                api_url=self.config.refine_api_url,
                api_key=self.config.refine_api_key,
                model_name=self.config.refine_model_name
            )
            
            if response and response.text():
                numbers = re.findall(r'0\.\d+|1\.0|0', response.text().strip())
                if numbers:
                    return min(float(numbers[0]), 1.0)
            
            return 0.5 # 默认值
            
        except Exception as e:
            logger.warning(f"LLM{analysis_name}计算失败，使用简化算法: {e}")
            return fallback_function(text)

    async def _calculate_formal_level(self, text: str) -> float:
        """使用LLM计算正式程度"""
        prompt_template = self.prompts.MULTIDIMENSIONAL_ANALYZER_FORMAL_LEVEL_PROMPT
        return await self._call_llm_for_style_analysis(text, prompt_template, self._simple_formal_level, "正式程度")

    def _simple_formal_level(self, text: str) -> float:
        """简化的正式程度计算（备用）"""
        formal_indicators = ['您', '请', '谢谢您', '不好意思', '打扰了', '恕我直言', '请问']
        informal_indicators = ['哈哈', '嘿', '啊', '呀', '哦', '嗯嗯', '哇']
        
        formal_count = sum(text.count(word) for word in formal_indicators)
        informal_count = sum(text.count(word) for word in informal_indicators)
        
        total = formal_count + informal_count
        return formal_count / max(total, 1) if total > 0 else 0.5

    async def _calculate_enthusiasm_level(self, text: str) -> float:
        """使用LLM计算热情程度"""
        prompt_template = self.prompts.MULTIDIMENSIONAL_ANALYZER_ENTHUSIASM_LEVEL_PROMPT
        return await self._call_llm_for_style_analysis(text, prompt_template, self._simple_enthusiasm_level, "热情程度")

    def _simple_enthusiasm_level(self, text: str) -> float:
        """简化的热情程度计算（备用）"""
        enthusiasm_indicators = ['！', '!', '哈哈', '太好了', '棒', '赞', '😄', '😊', '🎉', '厉害', 'awesome']
        count = sum(text.count(indicator) for indicator in enthusiasm_indicators)
        return min(count / max(len(text), 1) * 20, 1.0)

    async def _calculate_question_tendency(self, text: str) -> float:
        """使用LLM计算提问倾向"""
        prompt_template = self.prompts.MULTIDIMENSIONAL_ANALYZER_QUESTION_TENDENCY_PROMPT
        return await self._call_llm_for_style_analysis(text, prompt_template, self._simple_question_tendency, "提问倾向")

    def _simple_question_tendency(self, text: str) -> float:
        """简化的提问倾向计算（备用）"""
        question_indicators = ['？', '?', '吗', '呢', '什么', '怎么', '为什么', '哪里', '如何']
        count = sum(text.count(indicator) for indicator in question_indicators)
        return min(count / max(len(text), 1) * 10, 1.0)

    def _calculate_emoji_usage(self, text: str) -> float:
        """计算表情符号使用程度"""
        emoji_count = emoji.emoji_count(text)
        return min(emoji_count / max(len(text), 1) * 10, 1.0)

    def _calculate_punctuation_style(self, text: str) -> float:
        """计算标点符号风格"""
        punctuation_count = len([c for c in text if c in '，。！？；：""''()（）'])
        return punctuation_count / max(len(text), 1)

    async def get_user_insights(self, qq_id: str) -> Dict[str, Any]:
        """使用LLM生成深度用户洞察"""
        if qq_id not in self.user_profiles:
            return {"error": "用户不存在"}
        
        profile = self.user_profiles[qq_id]
        
        # 计算活跃时段
        active_hours = []
        if 'activity_hours' in profile.activity_pattern:
            sorted_hours = sorted(profile.activity_pattern['activity_hours'].items(), 
                                key=lambda x: x[1], reverse=True) # 修正排序键
            active_hours = [hour for hour, count in sorted_hours[:3]]
        
        # 计算主要话题
        main_topics = sorted(profile.topic_preferences.items(), 
                           key=lambda x: x[1], reverse=True)[:3] # 修正排序键
        
        # 计算社交活跃度
        social_activity = len(self.social_graph.get(qq_id, []))
        
        # 使用LLM生成深度洞察
        deep_insights = await self._generate_deep_insights(profile)
        
        return {
            'user_id': qq_id,
            'user_name': profile.qq_name,
            'nicknames': profile.nicknames,
            'active_hours': active_hours,
            'main_topics': [topic for topic, count in main_topics],
            'social_activity': social_activity,
            'communication_style_summary': self._summarize_communication_style(profile),
            'activity_summary': self._summarize_activity_pattern(profile),
            'deep_insights': deep_insights,
            'personality_analysis': await self._analyze_personality_traits(profile),
            'social_behavior': await self._analyze_social_behavior(qq_id)
        }

    async def _generate_deep_insights(self, profile: UserProfile) -> Dict[str, Any]:
        """使用LLM生成深度用户洞察"""
        if not (self.config.refine_api_url and self.config.refine_api_key and self.config.refine_model_name):
            logger.warning("提炼模型LLM客户端未初始化，无法使用LLM生成深度用户洞察。")
            return {"error": "LLM服务不可用"}

        try:
            # 准备用户数据摘要
            user_data_summary = {
                'qq_name': profile.qq_name,
                'nicknames': profile.nicknames,
                'topic_preferences': dict(list(profile.topic_preferences.items())[:5]),
                'activity_pattern': {
                    'peak_hours': [k for k, v in sorted(
                        profile.activity_pattern.get('activity_hours', {}).items(),
                        key=lambda item: item[1], reverse=True
                    )[:3]],
                    'avg_message_length': sum(profile.activity_pattern.get('message_lengths', [])) / 
                                        max(len(profile.activity_pattern.get('message_lengths', [])), 1)
                },
                'social_connections': len(profile.social_connections)
            }
            
            prompt = self.prompts.MULTIDIMENSIONAL_ANALYZER_DEEP_INSIGHTS_PROMPT.format(
                user_data_summary=json.dumps(user_data_summary, ensure_ascii=False, indent=2)
            )
            
            response = await self.refine_llm_client.chat_completion(
                prompt=prompt,
                api_url=self.config.refine_api_url,
                api_key=self.config.refine_api_key,
                model_name=self.config.refine_model_name
            )
            
            if response and response.text():
                try:
                    insights = json.loads(response.text().strip())
                    return insights
                except json.JSONDecodeError:
                    logger.warning(f"LLM响应JSON解析失败，返回简化分析。响应内容: {response.text()}")
                    return {
                        "personality_type": "分析中",
                        "communication_preference": "待深入分析",
                        "social_role": "群体成员",
                        "learning_potential": 0.7
                    }
            return {"error": "LLM未返回有效响应"}
                
        except Exception as e:
            logger.warning(f"深度洞察生成失败: {e}")
            return {"error": "洞察生成失败"}

    async def _analyze_personality_traits(self, profile: UserProfile) -> Dict[str, float]:
        """分析用户人格特质"""
        if not (self.config.refine_api_url and self.config.refine_api_key and self.config.refine_model_name):
            logger.warning("提炼模型LLM客户端未初始化，无法使用LLM分析人格特质，使用简化算法。")
            return self._simple_personality_analysis(profile)

        try:
            # 获取最近的沟通风格数据
            recent_styles = {}
            for feature, values in profile.communication_style.items():
                if values:
                    recent_styles[feature] = sum(values[-10:]) / min(len(values), 10)
            
            prompt = self.prompts.MULTIDIMENSIONAL_ANALYZER_PERSONALITY_TRAITS_PROMPT.format(
                communication_style_data=json.dumps(recent_styles, ensure_ascii=False, indent=2)
            )
            
            response = await self.refine_llm_client.chat_completion(
                prompt=prompt,
                api_url=self.config.refine_api_url,
                api_key=self.config.refine_api_key,
                model_name=self.config.refine_model_name
            )
            
            if response and response.text():
                try:
                    traits = json.loads(response.text().strip())
                    return traits
                except json.JSONDecodeError:
                    logger.warning(f"LLM响应JSON解析失败，返回简化人格分析。响应内容: {response.text()}")
                    return self._simple_personality_analysis(profile)
            return self._simple_personality_analysis(profile)
                
        except Exception as e:
            logger.warning(f"人格特质分析失败: {e}")
            return self._simple_personality_analysis(profile)

    def _simple_personality_analysis(self, profile: UserProfile) -> Dict[str, float]:
        """简化的人格分析（备用）"""
        # 基于基础数据的简单分析
        style_data = profile.communication_style
        
        # 外向性：基于消息频率和长度
        extraversion = 0.5
        if 'length' in style_data and style_data['length']:
            avg_length = sum(style_data['length'][-20:]) / min(len(style_data['length']), 20)
            extraversion = min(avg_length / 100, 1.0)
        
        # 开放性：基于话题多样性
        openness = len(profile.topic_preferences) / 10 if profile.topic_preferences else 0.5
        
        return {
            "openness": min(openness, 1.0),
            "conscientiousness": 0.6,  # 默认值
            "extraversion": extraversion,
            "agreeableness": 0.7,  # 默认值
            "neuroticism": 0.3   # 默认值
        }

    async def _analyze_social_behavior(self, qq_id: str) -> Dict[str, Any]:
        """分析社交行为模式"""
        if qq_id not in self.social_graph:
            return {"interaction_count": 0, "relationship_strength": {}}
        
        relations = self.social_graph[qq_id]
        
        # 统计不同类型的社交行为
        behavior_stats = {
            "mention_frequency": len([r for r in relations if r.relation_type == 'mention']),
            "reply_frequency": len([r for r in relations if r.relation_type == 'reply']),
            "total_interactions": len(relations),
            "avg_relationship_strength": sum(r.strength for r in relations) / max(len(relations), 1),
            "top_connections": [
                {"user": r.to_user, "strength": r.strength, "frequency": r.frequency}
                for r in sorted(relations, key=lambda x: x.strength, reverse=True)[:5]
            ]
        }
        
        return behavior_stats

    def _summarize_communication_style(self, profile: UserProfile) -> Dict[str, str]:
        """总结沟通风格"""
        style_summary = {}
        
        if 'length' in profile.communication_style and profile.communication_style['length']:
            avg_length = sum(profile.communication_style['length']) / len(profile.communication_style['length'])
            if avg_length > 50:
                style_summary['length_style'] = '详细型'
            elif avg_length > 20:
                style_summary['length_style'] = '适中型'
            else:
                style_summary['length_style'] = '简洁型'
        
        return style_summary

    def _summarize_activity_pattern(self, profile: UserProfile) -> Dict[str, Any]:
        """总结活动模式"""
        activity_summary = {}
        
        if 'activity_hours' in profile.activity_pattern:
            hours = profile.activity_pattern['activity_hours']
            if hours:
                peak_hour = max(hours.items(), key=lambda x: x[1])[0] # 修正为获取键
                activity_summary['peak_hour'] = peak_hour
                activity_summary['peak_period'] = self._get_time_period(peak_hour)
        
        return activity_summary

    async def export_social_graph(self) -> Dict[str, Any]:
        """导出社交关系图谱"""
        graph_data = {
            'nodes': [],
            'edges': [],
            'statistics': {}
        }
        
        # 导出节点（用户）
        # 从数据库加载所有用户画像，而不是只���内存中获取
        # 为了简化，这里仍然使用内存中的 user_profiles，但实际应该从数据库加载
        for qq_id, profile in self.user_profiles.items():
            graph_data['nodes'].append({
                'id': qq_id,
                'name': profile.qq_name,
                'nicknames': profile.nicknames,
                'activity_level': len(profile.activity_pattern.get('activity_hours', {}))
            })
        
        # 导出边（关系）
        # 从数据库加载所有社交关系，而不是只从内存中获取
        # 为了简化，这里仍然使用内存中的 social_graph，但实际应该从数据库加载
        for from_user, relations in self.social_graph.items():
            for relation in relations:
                graph_data['edges'].append({
                    'from': from_user,
                    'to': relation.to_user,
                    'type': relation.relation_type,
                    'strength': relation.strength,
                    'frequency': relation.frequency
                })
        
        # 统计信息
        graph_data['statistics'] = {
            'total_users': len(self.user_profiles),
            'total_relations': sum(len(relations) for relations in self.social_graph.values()),
            'nickname_mappings': len(self.nickname_mapping)
        }
        
        return graph_data
