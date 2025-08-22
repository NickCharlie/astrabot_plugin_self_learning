"""
智能回复器 - 调用AstrBot框架发送增强的智能回复
"""
import json
import time
import random # 移动到文件顶部
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from astrbot.api import logger
from astrbot.api.star import Context
from astrbot.api.event import AstrMessageEvent
from astrbot.core.platform.message_type import MessageType

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
    
    def __init__(self, config: PluginConfig, context: Context, db_manager, llm_client, prompts: Any):
        self.config = config
        self.context = context
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.prompts = prompts
        
        # 设置默认回复策略 - 不依赖配置文件
        self.enable_intelligent_reply = True  # 默认启用智能回复
        self.context_window_size = 5  # 默认上下文窗口大小
        
        logger.info("智能回复器初始化完成 - 使用默认配置")

    async def should_respond(self, event: AstrMessageEvent) -> bool:
        """判断是否应该回复此消息"""
        if not self.enable_intelligent_reply:
            return False
        
        try:
            # 获取消息类型 (私聊或群聊)
            group_id = event.get_group_id()
            is_group_chat = True if event.get_message_type() == MessageType.GROUP_MESSAGE else False
            is_private_chat = not is_group_chat
            message_text = event.get_message_str()
            
            if is_private_chat:
                # 私聊消息一定回复
                logger.debug(f"私聊消息，将回复: {message_text[:50]}")
                return True
            elif is_group_chat:
                # 群聊消息只有被 @ 或唤醒时才回复
                if hasattr(event, 'is_at_or_wake_command') and event.is_at_or_wake_command:
                    logger.debug(f"群聊消息被@或唤醒，将回复: {message_text[:50]}")
                    return True
                else:
                    logger.debug(f"群聊消息未被@或唤醒，不回复: {message_text[:50]}")
                    return False
            
            return False # 默认不回复
            
        except Exception as e:
            logger.error(f"判断是否回复失败: {e}")
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

    async def generate_intelligent_response_text(self, event: AstrMessageEvent) -> Optional[str]:
        """生成自学习可能需要用到的的智能回复文本（原来的逻辑）"""
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
            
            # 使用框架当前的人格设定作为系统提示词
            provider = self.context.get_using_provider()
            system_prompt = None
            if provider and hasattr(provider, 'curr_personality') and provider.curr_personality:
                system_prompt = provider.curr_personality.get('prompt', '你是一个友好、智能的助手。')
            else:
                system_prompt = '你是一个友好、智能的助手。'
            
            # 使用传入的 llm_client 进行聊天补全
            response = await self.llm_client.chat_completion(
                api_url=self.config.refine_api_url or "https://api.openai.com/v1/chat/completions", 
                api_key=self.config.refine_api_key or "", 
                model_name=self.config.refine_model_name or "gpt-4o", 
                prompt=enhanced_prompt,
                system_prompt=system_prompt,  # 使用框架的人格设定
                temperature=0.7,  # 使用默认温度
                max_tokens=self.PROMPT_RESPONSE_WORD_LIMIT
            )
            
            if response and response.text():
                response_text = response.text()
                # 记录回复
                await self._record_response(group_id, sender_id, message_text, response_text)
                return response_text.strip()
            else:
                logger.warning("LLM 未返回有效回复。")
                return None
            
        except Exception as e:
            logger.error(f"生成智能回复文本失败: {e}")
            raise ResponseError(f"生成智能回复文本失败: {str(e)}")

    async def generate_intelligent_response(self, event: AstrMessageEvent) -> Optional[Dict[str, Any]]:
        """生成智能回复参数，用于传递给框架的request_llm"""
        try:
            sender_id = event.get_sender_id()
            group_id = event.get_group_id()
            message_text = event.get_message_str()
            
            # 收集上下文信息
            context_info = await self._collect_context_info(group_id, sender_id, message_text)
            
            # 构建增强提示词，包含所有人格增量更新和社交关系信息
            enhanced_prompt = await self._build_enhanced_prompt(context_info, message_text)
            
            # 获取当前会话信息
            conversation = await self._get_conversation_context(group_id, sender_id)
            
            # 获取当前会话ID
            curr_cid = f"{group_id}_{sender_id}" if group_id else sender_id
            
            # 返回request_llm所需的参数
            return {
                'prompt': enhanced_prompt,
                'session_id': curr_cid,
                'conversation': conversation
            }
            
        except Exception as e:
            logger.error(f"生成智能回复参数失败: {e}")
            raise ResponseError(f"生成智能回复参数失败: {str(e)}")

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
        """构建增强的提示词，包含人格增量更新和社交关系等信息"""
        prompt_parts = []
        
        # 1. 基础场景设定
        prompt_parts.append("你正在参与一个真实的群聊对话，需要基于以下详细上下文信息进行自然、智能的回复：")
        
        # 2. 当前人格状态 - 获取完整的人格信息（包含增量更新）
        provider = self.context.get_using_provider()
        current_persona = "你是一个友好、智能的助手。"  # 默认人格
        persona_updates_info = ""
        
        if provider and hasattr(provider, 'curr_personality') and provider.curr_personality:
            current_persona = provider.curr_personality.get('prompt', current_persona)
            
            # 检查并提取增量更新信息
            if "【增量更新" in current_persona:
                # 提取所有增量更新部分
                import re
                update_pattern = r'【增量更新[^】]*】[^【]*'
                updates = re.findall(update_pattern, current_persona)
                if updates:
                    persona_updates_info = f"\n\n【当前活跃的人格增量更新】:\n" + "\n".join(updates[-3:])  # 取最近3个更新
            
            logger.debug(f"获取到当前人格设定长度: {len(current_persona)} 字符")
        
        prompt_parts.append(f"""
        【人格设定】:
        {current_persona}
        {persona_updates_info}
        """)
        
        # 3. 用户画像信息（详细展示）
        if context_info['sender_profile']:
            profile = context_info['sender_profile']
            prompt_parts.append(f"""
            【用户画像】:
            - 用户ID: {profile.get('qq_id', '未知')}
            - 昵称: {profile.get('qq_name', '未知')}
            - 沟通风格: {json.dumps(profile.get('communication_style', {}), ensure_ascii=False)}
            - 话题偏好: {json.dumps(profile.get('topic_preferences', {}), ensure_ascii=False)}
            - 情感倾向: {profile.get('emotional_tendency', '未知')}
            - 活跃时段: {profile.get('active_hours', '未知')}
            """)
        
        # 4. 社交关系图谱（增强版）
        if context_info['social_relations']:
            relations_details = []
            for rel in context_info['social_relations'][:5]:  # 显示前5个关系
                strength_desc = "强" if rel['strength'] > 0.7 else "中" if rel['strength'] > 0.4 else "弱"
                relations_details.append(
                    f"- 与{rel.get('to_user', '未知用户')}的关系强度: {rel['strength']:.2f}({strength_desc}), "
                    f"互动次数: {rel.get('interaction_count', 0)}, "
                    f"关系类型: {rel.get('relation_type', '普通')}"
                )
            
            prompt_parts.append(f"""
            【社交关系图谱】:
            {chr(10).join(relations_details)}
            """)
        
        # 5. 群聊氛围和活跃度分析
        atmosphere = context_info['group_atmosphere']
        activity_desc = "高度活跃" if atmosphere.get('activity_level') == 'high' else "一般活跃"
        prompt_parts.append(f"""
        【群聊环境】:
        - 当前活跃度: {activity_desc}
        - 平均消息长度: {atmosphere.get('avg_message_length', 0):.1f}字符
        - 最近消息数: {atmosphere.get('total_recent_messages', 0)}条
        - 群聊氛围: {"热烈讨论" if atmosphere.get('total_recent_messages', 0) > 10 else "轻松聊天"}
        """)
        
        # 6. 最近对话上下文（更详细）
        if context_info['recent_messages']:
            recent_context = []
            for i, msg in enumerate(context_info['recent_messages'][-5:], 1):  # 最近5条
                quality_score = msg.get('quality_scores', {})
                msg_quality = "高质量" if isinstance(quality_score, dict) and quality_score.get('overall', 0) > 0.7 else "普通"
                recent_context.append(
                    f"{i}. {msg.get('sender_name', '未知')}: {msg['message'][:80]}{'...' if len(msg['message']) > 80 else ''} "
                    f"(消息质量: {msg_quality})"
                )
            
            prompt_parts.append(f"""
            【最近对话上下文】:
            {chr(10).join(recent_context)}
            """)
        
        # 7. 时间和情境信息
        time_context = context_info.get('time_context', datetime.now().isoformat())
        hour = datetime.now().hour
        time_period = "早上" if 6 <= hour < 12 else "下午" if 12 <= hour < 18 else "晚上" if 18 <= hour < 22 else "深夜"
        
        prompt_parts.append(f"""
        【时间情境】:
        - 当前时间: {time_context[:16]}
        - 时段: {time_period}
        - 建议语气: {"活力充沛" if time_period in ["早上", "下午"] else "温和轻松"}
        """)
        
        # 8. 当前用户消息
        prompt_parts.append(f"""
        【当前用户消息】: {message}
        """)
        
        # 9. 回复指导原则（增强版）
        prompt_parts.append(f"""
        【回复要求】:
        1. 严格按照人格设定进行回复，特别注意增量更新的特征
        2. 根据用户画像调整回复风格和内容偏好
        3. 考虑社交关系强度，对关系较强的用户更加亲近
        4. 适应当前群聊氛围和活跃度
        5. 参考最近对话上下文，保持话题连贯性
        6. 根据时间情境调整语气和活跃度
        7. 回复要自然流畅，长度控制在{self.PROMPT_RESPONSE_WORD_LIMIT}字以内
        8. 避免重复性回复，体现个性化和智能化
        9. 如果用户表达情感，要给予适当的情感回应
        10. 保持角色一致性，不要出戏
        """)
        
        return "\n".join(prompt_parts)

    async def _get_conversation_context(self, group_id: str, sender_id: str) -> List[Dict[str, str]]:
        """获取对话上下文"""
        try:
            # 获取最近的消息作为对话上下文
            recent_messages = await self.db_manager.get_recent_filtered_messages(group_id, self.context_window_size)
            
            conversation = []
            for msg in recent_messages:
                # 将消息转换为对话格式
                conversation.append({
                    "role": "user" if msg['sender_id'] != "bot" else "assistant",
                    "content": msg['message']
                })
            
            return conversation
            
        except Exception as e:
            logger.error(f"获取对话上下文失败: {e}")
            return []

    async def _record_response(self, group_id: str, sender_id: str, original_message: str, response: str):
        """记录回复信息用于学习"""
        try:
            conn = await self.db_manager._get_messages_db_connection()
            cursor = await conn.cursor()
            
            # 简化实现：filtered_messages 表用于记录所有经过筛选的消息，包括BOT的回复。
            # 实际应用中，可能需要为BOT回复创建单独的表以区分。
            await cursor.execute('''
                INSERT OR IGNORE INTO filtered_messages 
                (message, sender_id, group_id, confidence, filter_reason, timestamp, used_for_learning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"BOT回复: {response}",
                "bot",
                group_id,  # 添加 group_id 字段
                1.0, # 假设BOT回复的置信度为1.0
                f"回复{sender_id}: {original_message[:self.PROMPT_MESSAGE_LENGTH_LIMIT]}", # 使用常量
                time.time(),
                False # BOT回复不用于学习，避免循环学习
            ))
            
            await conn.commit()
            
        except Exception as e:
            logger.error(f"记录回复失败: {e}")

    async def send_intelligent_response(self, event: AstrMessageEvent):
        """发送智能回复 - 返回request_llm参数供main.py使用yield发送"""
        try:
            if not await self.should_respond(event):
                return None
            
            response_params = await self.generate_intelligent_response(event)
            
            if response_params:
                logger.info(f"生成智能回复参数: prompt长度={len(response_params['prompt'])}字符, session_id={response_params['session_id']}")
                return response_params  # 返回request_llm参数
            else:
                return None
                
        except Exception as e:
            logger.error(f"生成智能回复参数失败: {e}")
            return None

    async def get_response_statistics(self, group_id: str) -> Dict[str, Any]:
        """获取回复统计"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = await conn.cursor()
            
            # 统计BOT回复次数
            await cursor.execute('''
                SELECT COUNT(*) 
                FROM filtered_messages 
                WHERE sender_id = 'bot' AND timestamp > ?
            ''', (time.time() - self.DAILY_RESPONSE_STATS_PERIOD_SECONDS,))  # 最近24小时
            
            row = await cursor.fetchone()
            daily_responses = row[0] if row else 0
            
            return {
                'daily_responses': daily_responses,
                'intelligent_reply_enabled': self.enable_intelligent_reply
            }
            
        except Exception as e:
            logger.error(f"获取回复统计失败: {e}")
            return {}

    async def _analyze_group_atmosphere(self, group_id: str) -> Dict[str, Any]:
        """分析群氛围"""
        try:
            # 从全局消息数据库获取连接
            conn = await self.db_manager._get_messages_db_connection()
            cursor = await conn.cursor()
            
            # 分析最近消息的情感倾向
            await cursor.execute('''
                SELECT COUNT(*) as total_messages,
                       AVG(LENGTH(message)) as avg_length
                FROM raw_messages 
                WHERE timestamp > ?
            ''', (time.time() - self.GROUP_ATMOSPHERE_PERIOD_SECONDS,))  # 最近1小时
            
            row = await cursor.fetchone()
            
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
