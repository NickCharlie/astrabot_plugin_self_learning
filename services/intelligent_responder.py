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

from ..core.framework_llm_adapter import FrameworkLLMAdapter  # 导入框架适配器
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
    
    def __init__(self, config: PluginConfig, context: Context, db_manager, 
                 llm_adapter: Optional[FrameworkLLMAdapter] = None,
                 prompts: Any = None, affection_manager = None):
        self.config = config
        self.context = context
        self.db_manager = db_manager
        self.prompts = prompts
        self.affection_manager = affection_manager  # 添加好感度管理器
        
        # 使用框架适配器
        self.llm_adapter = llm_adapter
        
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
        """生成自学习可能需要用到的的智能回复文本（修改版 - 增量更新在SYSTEM_PROMPT中）"""
        try:
            sender_id = event.get_sender_id()
            group_id = event.get_group_id() or event.get_sender_id()  # 私聊时使用 sender_id 作为会话 ID
            message_text = event.get_message_str()
            
            # 收集上下文信息
            context_info = await self._collect_context_info(group_id, sender_id, message_text)
            
            # 获取基础系统提示词并增强
            enhanced_system_prompt = await self._build_enhanced_system_prompt(context_info)
            
            logger.debug(f"构建的增强系统提示词长度: {len(enhanced_system_prompt)} 字符")
            
            # 调用框架的默认LLM
            provider = self.context.get_using_provider()
            if not provider:
                logger.warning("未找到可用的LLM提供商")
                return None
            
            # 使用框架适配器
            if self.llm_adapter and self.llm_adapter.has_refine_provider():
                try:
                    response = await self.llm_adapter.refine_chat_completion(
                        prompt=message_text,  # 用户消息只包含原始消息
                        system_prompt=enhanced_system_prompt,  # 原有PROMPT + 增量更新
                        temperature=0.7,
                        max_tokens=self.PROMPT_RESPONSE_WORD_LIMIT
                    )
                    
                    if response:
                        response_text = response.strip()
                        # 记录回复
                        await self._record_response(group_id, sender_id, message_text, response_text)
                        return response_text
                    else:
                        logger.warning("框架适配器未返回有效回复。")
                        return None
                except Exception as e:
                    logger.error(f"框架适配器生成回复失败: {e}")
                    return None
            else:
                logger.warning("没有可用的LLM服务")
                return None
            
        except Exception as e:
            logger.error(f"生成智能回复文本失败: {e}")
            raise ResponseError(f"生成智能回复文本失败: {str(e)}")

    async def generate_intelligent_response(self, event: AstrMessageEvent) -> Optional[Dict[str, Any]]:
        """生成智能回复参数，用于传递给框架的request_llm"""
        try:
            sender_id = event.get_sender_id()
            group_id = event.get_group_id() or event.get_sender_id()  # 私聊时使用 sender_id 作为会话 ID
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
            'group_id': group_id,  # 添加group_id字段
            'sender_id': sender_id,  # 添加sender_id字段
            'sender_profile': None,
            'user_affection': None,
            'social_relations': [],
            'recent_messages': [],
            'group_atmosphere': {},
            'time_context': datetime.now().isoformat()
        }
        
        try:
            # 获取发送者画像
            context_info['sender_profile'] = await self.db_manager.load_user_profile(group_id, sender_id)
            
            # 获取用户好感度信息
            context_info['user_affection'] = await self.db_manager.get_user_affection(group_id, sender_id)
            
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

    async def _build_enhanced_system_prompt(self, context_info: Dict[str, Any]) -> str:
        """
        构建增强的系统提示词 = 原有PROMPT + 增量更新 + 用户上下文信息
        
        Args:
            context_info: 用户上下文信息
            
        Returns:
            str: 增强后的系统提示词
        """
        try:
            # 1. 获取基础人格设定（原有的SYSTEM_PROMPT）
            provider = self.context.get_using_provider()
            base_system_prompt = "你是一个友好、智能的助手。"  # 默认
            
            if provider and hasattr(provider, 'curr_personality') and provider.curr_personality:
                base_system_prompt = provider.curr_personality.get('prompt', base_system_prompt)
            
            logger.debug(f"原有系统提示词长度: {len(base_system_prompt)} 字符")
            
            # 2. 构建增量更新部分
            incremental_updates = ""
            
            # 检查是否已经包含增量更新（避免重复添加）
            if "【增量更新" not in base_system_prompt and "【当前情绪状态" not in base_system_prompt:
                # 从temporary_persona_updater获取当前的增量更新
                # 这里可以添加逻辑来获取最新的增量更新内容
                pass
            
            # 3. 构建用户上下文增强信息
            context_enhancement = await self._build_context_enhancement(context_info)
            
            # 4. 集成心情信息（如果好感度管理器可用）
            mood_enhanced_prompt = base_system_prompt
            if self.affection_manager:
                try:
                    # 从context_info获取group_id
                    group_id = context_info.get('group_id')
                    if group_id:
                        mood_enhanced_prompt = await self.affection_manager.get_mood_influenced_system_prompt(
                            group_id, base_system_prompt
                        )
                        logger.debug(f"心情系统已应用到系统提示词: {group_id}")
                except Exception as e:
                    logger.warning(f"应用心情系统提示词失败: {e}")
                    mood_enhanced_prompt = base_system_prompt
            
            # 5. 组合最终的系统提示词: 心情增强PROMPT + 增量更新 + 上下文增强
            enhanced_prompt = mood_enhanced_prompt
            
            if incremental_updates:
                enhanced_prompt += f"\n\n{incremental_updates}"
            
            if context_enhancement:
                enhanced_prompt += f"\n\n{context_enhancement}"
            
            logger.debug(f"增强后系统提示词长度: {len(enhanced_prompt)} 字符")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"构建增强系统提示词失败: {e}")
            # 返回基础提示词作为后备
            return "你是一个友好、智能的助手。"
    
    async def _build_context_enhancement(self, context_info: Dict[str, Any]) -> str:
        """
        构建用户上下文增强信息（添加到系统提示词末尾）
        
        Args:
            context_info: 用户上下文信息
            
        Returns:
            str: 上下文增强信息
        """
        try:
            enhancement_parts = []
            
            # 1. 获取发送者ID用于社交关系查询
            sender_id = context_info.get('sender_id', '')
            
            # 2. 用户画像信息（详细展示）
            if context_info.get('sender_profile'):
                profile = context_info['sender_profile']
                
                # 构建用户画像基础信息
                user_info_parts = [
                    f"- 用户ID: {profile.get('qq_id', '未知')}",
                    f"- 昵称: {profile.get('qq_name', '未知')}",
                    f"- 沟通风格: {json.dumps(profile.get('communication_style', {}), ensure_ascii=False)}",
                    f"- 话题偏好: {json.dumps(profile.get('topic_preferences', {}), ensure_ascii=False)}",
                    f"- 情感倾向: {profile.get('emotional_tendency', '未知')}",
                    f"- 活跃时段: {profile.get('active_hours', '未知')}"
                ]
                
                # 添加好感度信息
                if context_info.get('user_affection'):
                    affection_data = context_info['user_affection']
                    affection_level = affection_data.get('affection_level', 0)
                    last_interaction = affection_data.get('last_interaction', 0)
                    interaction_count = affection_data.get('interaction_count', 0)
                    
                    # 计算好感度等级和描述
                    if affection_level >= 80:
                        affection_desc = "非常亲密"
                    elif affection_level >= 60:
                        affection_desc = "关系良好"
                    elif affection_level >= 40:
                        affection_desc = "较为熟悉"
                    elif affection_level >= 20:
                        affection_desc = "初步认识"
                    else:
                        affection_desc = "刚认识"
                    
                    # 计算交互频率描述
                    import time
                    days_since_last = (time.time() - last_interaction) / 86400 if last_interaction > 0 else 999
                    if days_since_last <= 1:
                        interaction_desc = "经常互动"
                    elif days_since_last <= 7:
                        interaction_desc = "偶尔互动"
                    else:
                        interaction_desc = "很少互动"
                    
                    user_info_parts.extend([
                        f"- 好感度: {affection_level}/100 ({affection_desc})",
                        f"- 交互次数: {interaction_count}次",
                        f"- 交互频率: {interaction_desc}"
                    ])
                else:
                    user_info_parts.append("- 好感度: 0/100 (新用户)")
                
                enhancement_parts.append(f"""
                【用户画像】:
                {chr(10).join(user_info_parts)}
                """)
            else:
                # 如果没有用户画像，至少显示好感度信息
                if context_info.get('user_affection'):
                    affection_data = context_info['user_affection']
                    affection_level = affection_data.get('affection_level', 0)
                    
                    if affection_level >= 80:
                        affection_desc = "非常亲密"
                    elif affection_level >= 60:
                        affection_desc = "关系良好"
                    elif affection_level >= 40:
                        affection_desc = "较为熟悉"
                    elif affection_level >= 20:
                        affection_desc = "初步认识"
                    else:
                        affection_desc = "刚认识"
                    
                    enhancement_parts.append(f"""
                    【用户信息】:
                    - 好感度: {affection_level}/100 ({affection_desc})
                    - 交互次数: {affection_data.get('interaction_count', 0)}次
                    """)
                else:
                    enhancement_parts.append("""
                    【用户信息】:
                    - 好感度: 0/100 (新用户)
                    - 交互次数: 0次
                    """)
            
            # 3. 社交关系图谱（增强版）
            if context_info.get('social_relations'):
                relations_details = []
                for rel in context_info['social_relations'][:5]:  # 显示前5个关系
                    strength_desc = "强" if rel['strength'] > 0.7 else "中" if rel['strength'] > 0.4 else "弱"
                    relations_details.append(
                        f"- 与{rel.get('to_user', '未知用户')}的关系强度: {rel['strength']:.2f}({strength_desc}), "
                        f"互动次数: {rel.get('interaction_count', 0)}, "
                        f"关系类型: {rel.get('relation_type', '普通')}"
                    )
                
                enhancement_parts.append(f"""
                【社交关系图谱】:
                {chr(10).join(relations_details)}
                """)
            
            # 4. 群聊氛围和活跃度分析
            atmosphere = context_info.get('group_atmosphere', {})
            activity_desc = "高度活跃" if atmosphere.get('activity_level') == 'high' else "一般活跃"
            enhancement_parts.append(f"""
            【群聊环境】:
            - 当前活跃度: {activity_desc}
            - 平均消息长度: {atmosphere.get('avg_message_length', 0):.1f}字符
            - 最近消息数: {atmosphere.get('total_recent_messages', 0)}条
            - 群聊氛围: {"热烈讨论" if atmosphere.get('total_recent_messages', 0) > 10 else "轻松聊天"}
            """)
            
            # 5. 最近对话上下文（更详细）
            if context_info.get('recent_messages'):
                recent_context = []
                for i, msg in enumerate(context_info['recent_messages'][-5:], 1):  # 最近5条
                    quality_score = msg.get('quality_scores', {})
                    msg_quality = "高质量" if isinstance(quality_score, dict) and quality_score.get('overall', 0) > 0.7 else "普通"
                    recent_context.append(
                        f"{i}. {msg.get('sender_name', '未知')}: {msg['message'][:80]}{'...' if len(msg['message']) > 80 else ''} "
                        f"(消息质量: {msg_quality})"
                    )
                
                enhancement_parts.append(f"""
                【最近对话上下文】:
                {chr(10).join(recent_context)}
                """)
            
            # 7. 回复指导原则（增强版）
            enhancement_parts.append(f"""
            【回复要求】:
            1. 根据用户画像调整回复风格和内容偏好
            2. **根据用户好感度调整亲密程度**：
            - 好感度0-20：保持礼貌但略显生疏，使用敬语
            - 好感度21-40：友好但不过分亲近，正常交流
            - 好感度41-60：较为熟悉的朋友语气，可以开玩笑
            - 好感度61-80：亲近朋友语气，更多关心和互动
            - 好感度81-100：非常亲密的关系，可以撒娇或使用昵称
            3. 考虑社交关系强度，对关系较强的用户更加亲近
            4. 适应当前群聊氛围和活跃度
            5. 参考最近对话上下文，保持话题连贯性
            7. 回复要自然流畅，长度控制在{self.PROMPT_RESPONSE_WORD_LIMIT}字以内
            8. 避免重复性回复，体现个性化和智能化
            9. 如果用户表达情感，要给予适当的情感回应
            10. 保持角色一致性，不要出戏
            11. 对于高好感度用户，可以主动关心和询问，体现更多人情味
            """)
            
            # 组合所有增强信息
            if enhancement_parts:
                return "\n\n".join(enhancement_parts)
            else:
                return ""
            
        except Exception as e:
            logger.error(f"构建用户上下文增强信息失败: {e}")
            return ""

    async def _build_enhanced_prompt(self, context_info: Dict[str, Any], message: str) -> str:
        """构建增强的提示词，包含人格增量更新和社交关系等信息"""
        try:
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
            
            # 3. 添加详细的上下文信息到提示词
            context_enhancement = await self._build_context_enhancement(context_info)
            if context_enhancement:
                prompt_parts.append(context_enhancement)
            
            # 4. 当前用户消息
            prompt_parts.append(f"""
            【当前用户消息】: {message}
            """)
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"构建增强提示词失败: {e}")
            return f"你是一个友好、智能的助手。请回复用户消息: {message}"

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
