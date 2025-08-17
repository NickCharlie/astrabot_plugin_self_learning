"""
智能回复器 - 调用AstrBot框架发送增强的智能回复
"""
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from astrbot.api import logger
from astrbot.api.star import Context
from astrbot.api.event import AstrMessageEvent

from ..config import PluginConfig
from ..exceptions import ResponseError


class IntelligentResponder:
    """智能回复器 - 基于用户画像和社交图谱生成智能回复"""

    # 常量定义
    SOCIAL_STRENGTH_THRESHOLD = 0.5
    REPLY_PROBABILITY_HIGH_SOCIAL = 0.3
    SOCIAL_RELATIONS_LIMIT = 5
    RECENT_MESSAGES_LIMIT = 5
    PROMPT_MESSAGE_LENGTH_LIMIT = 50
    PROMPT_RESPONSE_WORD_LIMIT = 100
    DAILY_RESPONSE_STATS_PERIOD_SECONDS = 86400  # 24小时
    GROUP_ATMOSPHERE_PERIOD_SECONDS = 3600  # 1小时
    GROUP_ACTIVITY_HIGH_THRESHOLD = 10
    
    def __init__(self, config: PluginConfig, context: Context, db_manager):
        self.config = config
        self.context = context
        self.db_manager = db_manager
        
        # 回复策略配置
        self.enable_intelligent_reply = config.enable_intelligent_reply
        self.reply_probability = config.reply_probability
        self.context_window_size = config.context_window_size
        
        logger.info("智能回复器初始化完成")

    async def should_respond(self, event: AstrMessageEvent) -> bool:
        """判断是否应该回复此消息"""
        if not self.enable_intelligent_reply:
            return False
        
        try:
            sender_id = event.get_sender_id()
            group_id = event.get_group_id()
            message_text = event.get_message_str()
            
            # 检查回复概率
            import random
            if random.random() > self.reply_probability:
                return False
            
            # 检查是否提到了BOT或有关键词
            if await self._is_relevant_message(message_text, group_id):
                return True
            
            # 检查社交关系强度
            social_strength = await self._get_social_strength(group_id, sender_id)
            if social_strength > self.SOCIAL_STRENGTH_THRESHOLD:  # 高社交强度用户更容易触发回复
                return random.random() < self.REPLY_PROBABILITY_HIGH_SOCIAL
            
            return False
            
        except Exception as e:
            logger.error(f"判断是否回复失败: {e}")
            return False

    async def _is_relevant_message(self, message: str, group_id: str) -> bool:
        """检查消息是否与BOT相关"""
        # 检查@消息
        if '@' in message:
            return True
        
        # 检查关键词
        message_lower = message.lower()
        for keyword in self.config.intelligent_reply_keywords:
            if keyword.lower() in message_lower:
                return True
        
        return False

    async def _get_social_strength(self, group_id: str, user_id: str) -> float:
        """获取用户的社交强度"""
        try:
            social_relations = await self.db_manager.load_social_graph(group_id)
            
            total_strength = 0.0
            relation_count = 0
            
            for relation in social_relations:
                if relation['from_user'] == user_id or relation['to_user'] == user_id:
                    total_strength += relation['strength']
                    relation_count += 1
            if relation_count == 0:
                return 0.0
            return total_strength / relation_count
            
        except Exception as e:
            logger.error(f"获取社交强度失败: {e}")
            return 0.0

    async def generate_intelligent_response(self, event: AstrMessageEvent) -> Optional[str]:
        """生成智能回复"""
        try:
            sender_id = event.get_sender_id()
            group_id = event.get_group_id()
            message_text = event.get_message_str()
            
            # 收集上下文信息
            context_info = await self._collect_context_info(group_id, sender_id, message_text)
            
            # 构建增强提示词
            enhanced_prompt = await self._build_enhanced_prompt(context_info, message_text)
            
            # 调用框架的默认LLM
            provider = self.context.get_using_provider()
            if not provider:
                logger.warning("未找到可用的LLM提供商")
                return None
            
            response = await provider.text_chat(enhanced_prompt)
            
            # 记录回复
            await self._record_response(group_id, sender_id, message_text, response)
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"生成智能回复失败: {e}")
            raise ResponseError(f"生成智能回复失败: {str(e)}")

    async def _collect_context_info(self, group_id: str, sender_id: str, message: str) -> Dict[str, Any]:
        """收集上下文信息"""
        context_info = {
            'sender_profile': None,
            'social_relations': [],
            'recent_messages': [],
            'group_atmosphere': {},
            'time_context': datetime.now().isoformat()
        }
        
        try:
            # 获取发送者画像
            context_info['sender_profile'] = await self.db_manager.load_user_profile(group_id, sender_id)
            
            # 获取相关社交关系
            all_relations = await self.db_manager.load_social_graph(group_id)
            context_info['social_relations'] = [
                rel for rel in all_relations 
                if rel['from_user'] == sender_id or rel['to_user'] == sender_id
            ][:5]  # 限制前5个最强关系
            
            # 获取最近的筛选消息
            context_info['recent_messages'] = await self.db_manager.get_recent_filtered_messages(group_id, 5)
            
            # 分析群氛围
            context_info['group_atmosphere'] = await self._analyze_group_atmosphere(group_id)
            
        except Exception as e:
            logger.error(f"收集上下文信息失败: {e}")
        
        return context_info

    async def _build_enhanced_prompt(self, context_info: Dict[str, Any], message: str) -> str:
        """构建增强的提示词"""
        prompt_parts = []
        
        # 基础人格设定
        current_persona = self.config.current_persona or "你是一个友好、智能的AI助手"
        prompt_parts.append(f"人格设定: {current_persona}")
        
        # 用户画像信息
        if context_info['sender_profile']:
            profile = context_info['sender_profile']
            prompt_parts.append(f"""
用户信息:
- QQ号: {profile.get('qq_id', '未知')}
- 昵称: {profile.get('qq_name', '未知')}
- 沟通风格: {json.dumps(profile.get('communication_style', {}), ensure_ascii=False)}
- 话题偏好: {json.dumps(profile.get('topic_preferences', {}), ensure_ascii=False)}
""")
        
        # 社交关系
        if context_info['social_relations']:
            relations_desc = []
            for rel in context_info['social_relations'][:3]:
                relations_desc.append(f"与{rel['to_user']}关系强度{rel['strength']:.2f}")
            prompt_parts.append(f"社交关系: {', '.join(relations_desc)}")
        
        # 群聊氛围
        atmosphere = context_info['group_atmosphere']
        prompt_parts.append(f"当前群聊氛围: {atmosphere.get('activity_level', '未知')}活跃度")
        
        # 最近消息上下文
        if context_info['recent_messages']:
            recent_msgs = []
            for msg in context_info['recent_messages']:
                recent_msgs.append(f"{msg['sender_name']}: {msg['message'][:50]} (评分: {msg['quality_scores']})")
            prompt_parts.append(f"最近对话: {' | '.join(recent_msgs)}")
        
        # 当前消息
        prompt_parts.append(f"当前消息: {message}")
        
        # 回复要求
        prompt_parts.append("""
请基于以上信息生成一个自然、智能的回复。要求:
1. 符合设定的人格特征
2. 考虑用户的沟通风格和偏好
3. 适应当前的群聊氛围
4. 回复简洁明了，不超过100字
5. 语言风格要自然流畅
""")
        
        return "\n\n".join(prompt_parts)

    async def _record_response(self, group_id: str, sender_id: str, original_message: str, response: str):
        """记录回复信息用于学习"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = conn.cursor()
            
            # 简化实现：filtered_messages 表用于记录所有经过筛选的消息，包括BOT的回复。
            # 实际应用中，可能需要为BOT回复创建单独的表以区分。
            cursor.execute('''
                INSERT OR IGNORE INTO filtered_messages 
                (message, sender_id, confidence, filter_reason, timestamp, used_for_learning)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"BOT回复: {response}",
                "bot",
                1.0, # 假设BOT回复的置信度为1.0
                f"回复{sender_id}: {original_message[:self.PROMPT_MESSAGE_LENGTH_LIMIT]}", # 使用常量
                time.time(),
                False # BOT回复不用于学习，避免循环学习
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"记录回复失败: {e}")

    async def send_intelligent_response(self, event: AstrMessageEvent):
        """发送智能回复"""
        try:
            if not await self.should_respond(event):
                return
            
            response = await self.generate_intelligent_response(event)
            
            if response:
                # 通过事件系统发送回复
                await event.send(response)
                logger.info(f"已发送智能回复: {response[:self.PROMPT_MESSAGE_LENGTH_LIMIT]}...") # 使用常量
                
        except Exception as e:
            logger.error(f"发送智能回复失败: {e}")

    async def get_response_statistics(self, group_id: str) -> Dict[str, Any]:
        """获取回复统计"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = conn.cursor()
            
            # 统计BOT回复次数
            cursor.execute('''
                SELECT COUNT(*) 
                FROM filtered_messages 
                WHERE sender_id = 'bot' AND timestamp > ?
            ''', (time.time() - self.DAILY_RESPONSE_STATS_PERIOD_SECONDS,))  # 最近24小时
            
            row = cursor.fetchone()
            daily_responses = row[0] if row else 0
            
            return {
                'daily_responses': daily_responses,
                'response_rate': self.reply_probability,
                'intelligent_reply_enabled': self.enable_intelligent_reply
            }
            
        except Exception as e:
            logger.error(f"获取回复统计失败: {e}")
            return {}
            
        except Exception as e:
            logger.error(f"获取最近消息失败: {e}")
            return []

    async def _analyze_group_atmosphere(self, group_id: str) -> Dict[str, Any]:
        """分析群聊氛围"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = conn.cursor()
            
            # 分析最近消息的情感倾向
            cursor.execute('''
                SELECT COUNT(*) as total_messages,
                       AVG(LENGTH(message)) as avg_length
                FROM raw_messages 
                WHERE timestamp > ?
            ''', (time.time() - self.GROUP_ATMOSPHERE_PERIOD_SECONDS,))  # 最近1小时
            
            row = cursor.fetchone()
            
            row = cursor.fetchone()
            total_messages = row[0] if row else 0
            avg_length = row[1] if row else 0.0
            
            return {
                'activity_level': 'high' if total_messages > self.GROUP_ACTIVITY_HIGH_THRESHOLD else 'low',
                'avg_message_length': avg_length,
                'total_recent_messages': total_messages
            }
            
        except Exception as e:
            logger.error(f"分析群氛围失败: {e}")
            return {'activity_level': 'unknown'}

    async def _build_enhanced_prompt(self, context_info: Dict[str, Any], message: str) -> str:
        """构建增强的提示词"""
        prompt_parts = []
        
        # 基础人格设定
        current_persona = self.config.current_persona or "你是一个友好、智能的AI助手"
        prompt_parts.append(f"人格设定: {current_persona}")
        
        # 用户画像信息
        if context_info['sender_profile']:
            profile = context_info['sender_profile']
            prompt_parts.append(f"""
用户信息:
- QQ号: {profile.get('qq_id', '未知')}
- 昵称: {profile.get('qq_name', '未知')}
- 沟通风格: {json.dumps(profile.get('communication_style', {}), ensure_ascii=False)}
- 话题偏好: {json.dumps(profile.get('topic_preferences', {}), ensure_ascii=False)}
""")
        
        # 社交关系
        if context_info['social_relations']:
            relations_desc = []
            for rel in context_info['social_relations'][:3]:
                relations_desc.append(f"与{rel['to_user']}关系强度{rel['strength']:.2f}")
            prompt_parts.append(f"社交关系: {', '.join(relations_desc)}")
        
        # 群聊氛围
        atmosphere = context_info['group_atmosphere']
        prompt_parts.append(f"当前群聊氛围: {atmosphere.get('activity_level', '未知')}活跃度")
        
        # 最近消息上下文
        if context_info['recent_messages']:
            recent_msgs = []
            for msg in context_info['recent_messages'][:3]:
                recent_msgs.append(f"{msg['sender_name']}: {msg['message'][:50]}")
            prompt_parts.append(f"最近对话: {' | '.join(recent_msgs)}")
        
        # 当前消息
        prompt_parts.append(f"当前消息: {message}")
        
        # 回复要求
        prompt_parts.append("""
请基于以上信息生成一个自然、智能的回复。要求:
1. 符合设定的人格特征
2. 考虑用户的沟通风格和偏好
3. 适应当前的群聊氛围
4. 回复简洁明了，不超过100字
5. 语言风格要自然流畅
""")
        
        return "\n\n".join(prompt_parts)

    async def _record_response(self, group_id: str, sender_id: str, original_message: str, response: str):
        """记录回复信息用于学习"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = conn.cursor()
            
            # 简化实现：可以创建一个回复记录表
            cursor.execute('''
                INSERT OR IGNORE INTO filtered_messages 
                (message, sender_id, confidence, filter_reason, timestamp, used_for_learning)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"BOT回复: {response}",
                "bot",
                1.0,
                f"回复{sender_id}: {original_message[:50]}",
                time.time(),
                False
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"记录回复失败: {e}")

    async def send_intelligent_response(self, event: AstrMessageEvent):
        """发送智能回复"""
        try:
            if not await self.should_respond(event):
                return
            
            response = await self.generate_intelligent_response(event)
            
            if response:
                # 通过事件系统发送回复
                await event.send(response)
                logger.info(f"已发送智能回复: {response[:50]}...")
                
        except Exception as e:
            logger.error(f"发送智能回复失败: {e}")

    async def get_response_statistics(self, group_id: str) -> Dict[str, Any]:
        """获取回复统计"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = conn.cursor()
            
            # 统计BOT回复次数
            cursor.execute('''
                SELECT COUNT(*) 
                FROM filtered_messages 
                WHERE sender_id = 'bot' AND timestamp > ?
            ''', (time.time() - self.DAILY_RESPONSE_STATS_PERIOD_SECONDS,))  # 最近24小时
            
            daily_responses = cursor.fetchone() if cursor.fetchone() else 0
            
            return {
                'daily_responses': daily_responses,
                'response_rate': self.reply_probability,
                'intelligent_reply_enabled': self.enable_intelligent_reply
            }
            
        except Exception as e:
            logger.error(f"获取回复统计失败: {e}")
            return {}
