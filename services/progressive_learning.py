"""
渐进式学习服务 - 协调各个组件实现智能自适应学习
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.star import Context

from ..config import PluginConfig
from ..exceptions import LearningError
from .database_manager import DatabaseManager # 导入 DatabaseManager
from .message_collector import MessageCollectorService
from .multidimensional_analyzer import MultidimensionalAnalyzer
from .style_analyzer import StyleAnalyzerService
from .learning_quality_monitor import LearningQualityMonitor


@dataclass
class LearningSession:
    """学习会话"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    messages_processed: int = 0
    filtered_messages: int = 0
    style_updates: int = 0
    quality_score: float = 0.0
    success: bool = False


class ProgressiveLearningService:
    """渐进式学习服务"""
    
    def __init__(self, config: PluginConfig, context: Context, 
                 message_collector: MessageCollectorService,
                 multidimensional_analyzer: MultidimensionalAnalyzer,
                 style_analyzer: StyleAnalyzerService,
                 quality_monitor: LearningQualityMonitor):
        self.config = config
        self.context = context
        self.db_manager: DatabaseManager = context.get_service("database_manager") # 获取 DatabaseManager 实例
        
        # 注入各个组件服务
        self.message_collector = message_collector
        self.multidimensional_analyzer = multidimensional_analyzer
        self.style_analyzer = style_analyzer
        self.quality_monitor = quality_monitor
        
        # 学习状态
        self.learning_active = False
        self.current_session: Optional[LearningSession] = None
        self.learning_sessions: List[LearningSession] = [] # 历史学习会话，可以从数据库加载
        
        # 学习控制参数
        self.batch_size = config.learning_batch_size
        self.learning_interval = config.learning_interval_minutes * 60  # 转换为秒
        self.quality_threshold = config.quality_threshold
        
        logger.info("渐进式学习服务初始化完成")

    async def start(self):
        """服务启动时加载历史学习会话"""
        # 假设每个群组有独立的学习会话，这里需要一个 group_id
        # 为了简化，暂时假设加载一个默认的或全局的学习会话
        # 实际应用中，可能需要根据当前处理的群组ID来加载
        default_group_id = "global_learning" # 或者从配置中获取
        # 这里可以加载所有历史会话，或者只加载最近的N个
        # 为了简化，我们暂时不从数据库加载历史会话列表，只在每次会话结束时保存
        # 如果需要加载历史会话，需要 DatabaseManager 提供 load_all_learning_sessions 方法
        logger.info("渐进式学习服务启动，准备开始学习。")

    async def start_learning(self, group_id: str) -> bool:
        """启动学习流程"""
        try:
            if self.learning_active:
                logger.warning("学习已经在进行中")
                return False
            
            self.learning_active = True
            
            # 创建新的学习会话
            session_id = f"session_{int(time.time())}"
            self.current_session = LearningSession(
                session_id=session_id,
                start_time=datetime.now().isoformat()
            )
            # 保存新的学习会话到数据库
            await self.db_manager.save_learning_session(group_id, self.current_session.__dict__)
            
            logger.info(f"开始学习会话: {session_id}")
            
            # 启动学习循环
            asyncio.create_task(self._learning_loop(group_id))
            
            return True
            
        except Exception as e:
            logger.error(f"启动学习失败: {e}")
            self.learning_active = False
            return False

    async def stop_learning(self):
        """停止学习流程"""
        self.learning_active = False
        
        if self.current_session:
            self.current_session.end_time = datetime.now().isoformat()
            self.current_session.success = True # 假设正常停止即成功
            # 保存更新后的学习会话到数据库
            default_group_id = "global_learning" # 或者从配置中获取
            await self.db_manager.save_learning_session(default_group_id, self.current_session.__dict__)
            self.learning_sessions.append(self.current_session) # 仍然添加到内存列表
            logger.info(f"学习会话结束: {self.current_session.session_id}")
            self.current_session = None

    async def _learning_loop(self, group_id: str):
        """主学习循环"""
        while self.learning_active:
            try:
                # 检查是否应该暂停学习
                should_pause, reason = await self.quality_monitor.should_pause_learning()
                if should_pause:
                    logger.warning(f"学习被暂停: {reason}")
                    await self.stop_learning()
                    break
                
                # 执行一个学习批次
                await self._execute_learning_batch(group_id)
                
                # 等待下一个学习周期
                await asyncio.sleep(self.learning_interval)
                
            except Exception as e:
                logger.error(f"学习循环异常: {e}")
                await asyncio.sleep(60)  # 异常时等待1分钟

    async def _execute_learning_batch(self, group_id: str):
        """执行一个学习批次"""
        try:
            batch_start_time = datetime.now()
            
            # 1. 获取未处理的消息
            unprocessed_messages = await self.message_collector.get_unprocessed_messages(
                limit=self.batch_size
            )
            
            if not unprocessed_messages:
                logger.debug("没有未处理的消息，跳过此批次")
                return
            
            logger.info(f"开始处理 {len(unprocessed_messages)} 条消息")
            
            # 2. 使用多维度分析器筛选消息
            filtered_messages = await self._filter_messages_with_context(unprocessed_messages)
            
            if not filtered_messages:
                logger.debug("没有通过筛选的消息")
                await self._mark_messages_processed(unprocessed_messages)
                return
            
            # 3. 使用风格分析器深度分析
            style_analysis = await self.style_analyzer.analyze_conversation_style(group_id, filtered_messages) # 传入 group_id
            
            # 4. 获取当前人格设置
            current_persona = await self._get_current_persona()
            
            # 5. 质量监控评估
            quality_metrics = await self.quality_monitor.evaluate_learning_batch(
                current_persona, 
                await self._generate_updated_persona(current_persona, style_analysis),
                filtered_messages
            )
            
            # 6. 根据质量评估决定是否应用更新
            if quality_metrics.consistency_score >= self.quality_threshold:
                await self._apply_learning_updates(style_analysis, filtered_messages)
                logger.info(f"学习更新已应用，质量得分: {quality_metrics.consistency_score:.3f}")
            else:
                logger.warning(f"学习质量不达标，跳过更新，得分: {quality_metrics.consistency_score:.3f}")
            
            # 7. 标记消息为已处理
            await self._mark_messages_processed(unprocessed_messages)
            
            # 8. 更新学习会话统计并持久化
            if self.current_session:
                self.current_session.messages_processed += len(unprocessed_messages)
                self.current_session.filtered_messages += len(filtered_messages)
                self.current_session.quality_score = quality_metrics.consistency_score
                # 每次批次结束都保存当前会话状态
                default_group_id = "global_learning" # 或者从配置中获取
                await self.db_manager.save_learning_session(default_group_id, self.current_session.__dict__)
            
            # 记录批次耗时
            batch_duration = (datetime.now() - batch_start_time).total_seconds()
            logger.info(f"学习批次完成，耗时: {batch_duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"学习批次执行失败: {e}")
            raise LearningError(f"学习批次执行失败: {str(e)}")

    # async def _execute_learning_batch(self):
    #     """执行一个学习批次"""
    #     try:
    #         batch_start_time = datetime.now()
            
    #         # 1. 获取未处理的消息
    #         unprocessed_messages = await self.message_collector.get_unprocessed_messages(
    #             limit=self.batch_size
    #         )
            
    #         if not unprocessed_messages:
    #             logger.debug("没有未处理的消息，跳过此批次")
    #             return
            
    #         logger.info(f"开始处理 {len(unprocessed_messages)} 条消息")
            
    #         # 2. 使用多维度分析器筛选消息
    #         filtered_messages = await self._filter_messages_with_context(unprocessed_messages)
            
    #         if not filtered_messages:
    #             logger.debug("没有通过筛选的消息")
    #             await self._mark_messages_processed(unprocessed_messages)
    #             return
            
    #         # 3. 使用风格分析器深度分析
    #         style_analysis = await self.style_analyzer.analyze_conversation_style(filtered_messages)
            
    #         # 4. 获取当前人格设置
    #         current_persona = await self._get_current_persona()
            
    #         # 5. 质量监控评估
    #         quality_metrics = await self.quality_monitor.evaluate_learning_batch(
    #             current_persona, 
    #             await self._generate_updated_persona(current_persona, style_analysis),
    #             filtered_messages
    #         )
            
    #         # 6. 根据质量评估决定是否应用更新
    #         if quality_metrics.consistency_score >= self.quality_threshold:
    #             await self._apply_learning_updates(style_analysis, filtered_messages)
    #             logger.info(f"学习更新已应用，质量得分: {quality_metrics.consistency_score:.3f}")
    #         else:
    #             logger.warning(f"学习质量不达标，跳过更新，得分: {quality_metrics.consistency_score:.3f}")
            
    #         # 7. 标记消息为已处理
    #         await self._mark_messages_processed(unprocessed_messages)
            
    #         # 8. 更新学习会话统计
    #         if self.current_session:
    #             self.current_session.messages_processed += len(unprocessed_messages)
    #             self.current_session.filtered_messages += len(filtered_messages)
    #             self.current_session.quality_score = quality_metrics.consistency_score
            
    #         # 记录批次耗时
    #         batch_duration = (datetime.now() - batch_start_time).total_seconds()
    #         logger.info(f"学习批次完成，耗时: {batch_duration:.2f}秒")
            
    #     except Exception as e:
    #         logger.error(f"学习批次执行失败: {e}")
    #         raise LearningError(f"学习批次执行失败: {str(e)}")

    async def _filter_messages_with_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用多维度分析进行智能筛选"""
        filtered = []
        
        for message in messages:
            try:
                # 创建模拟的事件对象进行分析
                context_analysis = await self.multidimensional_analyzer.analyze_message_context(
                    None,  # 简化实现，实际应传入真实事件
                    message['message']
                )
                
                # 根据上下文相关性筛选
                relevance = context_analysis.get('contextual_relevance', 0.0)
                if relevance >= self.config.relevance_threshold:
                    # 添加筛选信息到消息
                    message['context_analysis'] = context_analysis
                    message['relevance_score'] = relevance
                    filtered.append(message)
                    
                    # 保存到筛选消息表
                    await self.message_collector.add_filtered_message({
                        'raw_message_id': message.get('id'),
                        'message': message['message'],
                        'sender_id': message.get('sender_id', ''),
                        'confidence': relevance,
                        'filter_reason': 'context_relevance',
                        'timestamp': message.get('timestamp', time.time())
                    })
                    
            except Exception as e:
                logger.warning(f"消息筛选失败: {e}")
                continue
        
        return filtered

    async def _get_current_persona(self) -> Dict[str, Any]:
        """获取当前人格设置"""
        try:
            # 这里应该调用AstrBot的人格获取API
            # 简化实现，返回默认结构
            return {
                'prompt': self.config.current_persona or "默认人格",
                'style_parameters': {},
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取当前人格失败: {e}")
            return {'prompt': '默认人格', 'style_parameters': {}}

    async def _generate_updated_persona(self, current_persona: Dict[str, Any], style_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成更新后的人格"""
        try:
            provider = self.context.get_using_provider()
            if not provider:
                return current_persona
            
            prompt = f"""
基于当前人格设定和风格分析结果，生成优化后的人格prompt：

当前人格：
{current_persona.get('prompt', '')}

风格分析结果：
{json.dumps(style_analysis, ensure_ascii=False, indent=2)}

请返回JSON格式的更新后人格：
{{
    "prompt": "优化后的人格描述",
    "style_parameters": {{
        "formality": 0.0-1.0,
        "enthusiasm": 0.0-1.0,
        "interactivity": 0.0-1.0
    }},
    "update_reason": "更新原因说明"
}}
"""
            
            response = await provider.text_chat(prompt)
            
            try:
                updated_persona = json.loads(response.strip())
                updated_persona['last_updated'] = datetime.now().isoformat()
                return updated_persona
            except json.JSONDecodeError:
                logger.warning("人格更新JSON解析失败，保持原有人格")
                return current_persona
                
        except Exception as e:
            logger.error(f"生成更新人格失败: {e}")
            return current_persona

    async def _apply_learning_updates(self, style_analysis: Dict[str, Any], messages: List[Dict[str, Any]]):
        """应用学习更新"""
        try:
            # 1. 更新人格prompt（这里需要调用AstrBot的API）
            # 简化实现，记录日志
            logger.info("应用人格更新")
            
            # 2. 添加对话样本到语言模仿列表（这里需要调用AstrBot的API）
            # 简化实现，记录日志
            sample_messages = [msg['message'] for msg in messages[:5]]  # 取前5条作为样本
            logger.info(f"添加 {len(sample_messages)} 条对话样本到模仿列表")
            
            # 3. 记录学习更新
            if self.current_session:
                self.current_session.style_updates += 1
            
        except Exception as e:
            logger.error(f"应用学习更新失败: {e}")

    async def _mark_messages_processed(self, messages: List[Dict[str, Any]]):
        """标记消息为已处理"""
        message_ids = [msg['id'] for msg in messages if 'id' in msg]
        if message_ids:
            await self.message_collector.mark_messages_processed(message_ids)

    async def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        return {
            'learning_active': self.learning_active,
            'current_session': self.current_session.__dict__ if self.current_session else None,
            'total_sessions': len(self.learning_sessions),
            'statistics': await self.message_collector.get_statistics(),
            'quality_report': await self.quality_monitor.get_quality_report(),
            'last_update': datetime.now().isoformat()
        }

    async def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        try:
            # 获取风格趋势
            style_trends = await self.style_analyzer.get_style_trends()
            
            # 获取用户分析（示例用户）
            user_insights = {}
            if self.multidimensional_analyzer.user_profiles:
                sample_user_id = list(self.multidimensional_analyzer.user_profiles.keys())
                user_insights = await self.multidimensional_analyzer.get_user_insights(sample_user_id)
            
            # 获取社交图谱
            social_graph = await self.multidimensional_analyzer.export_social_graph()
            
            return {
                'style_trends': style_trends,
                'user_insights_sample': user_insights,
                'social_graph_summary': {
                    'total_nodes': len(social_graph.get('nodes', [])),
                    'total_edges': len(social_graph.get('edges', [])),
                    'statistics': social_graph.get('statistics', {})
                },
                'learning_performance': {
                    'successful_sessions': len([s for s in self.learning_sessions if s.success]),
                    'average_quality_score': sum(s.quality_score for s in self.learning_sessions) / 
                                           max(len(self.learning_sessions), 1),
                    'total_messages_processed': sum(s.messages_processed for s in self.learning_sessions)
                }
            }
            
        except Exception as e:
            logger.error(f"获取学习洞察失败: {e}")
            return {"error": str(e)}
