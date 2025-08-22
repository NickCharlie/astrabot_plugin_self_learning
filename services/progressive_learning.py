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
from .persona_manager import PersonaManagerService # 导入 PersonaManagerService
from ..utils.json_utils import safe_parse_llm_json


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
                 db_manager: DatabaseManager,
                 message_collector: MessageCollectorService,
                 multidimensional_analyzer: MultidimensionalAnalyzer,
                 style_analyzer: StyleAnalyzerService,
                 quality_monitor: LearningQualityMonitor,
                 persona_manager: PersonaManagerService, # 添加 persona_manager 参数
                 prompts: Any): # 添加 prompts 参数
        self.config = config
        self.context = context
        self.db_manager = db_manager
        
        # 注入各个组件服务
        self.message_collector = message_collector
        self.multidimensional_analyzer = multidimensional_analyzer
        self.style_analyzer = style_analyzer
        self.quality_monitor = quality_monitor
        self.persona_manager = persona_manager # 注入 persona_manager
        self.prompts = prompts  # 保存 prompts 实例
        
        # 学习状态 - 使用字典管理每个群组的学习状态
        self.learning_active = {}  # 改为字典，按群组ID管理
        self.current_session: Optional[LearningSession] = None
        self.learning_sessions: List[LearningSession] = [] # 历史学习会话，可以从数据库加载
        self.learning_lock = asyncio.Lock()  # 添加异步锁防止竞态条件
        
        # 学习控制参数
        self.batch_size = config.max_messages_per_batch
        self.learning_interval = config.learning_interval_hours * 3600  # 转换为秒
        self.quality_threshold = config.style_update_threshold
        
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
        """启动学习流程 - 优化为后台任务执行"""
        async with self.learning_lock:  # 使用锁防止竞态条件
            try:
                # 检查该群组是否已经在学习
                if self.learning_active.get(group_id, False):
                    logger.info(f"群组 {group_id} 学习已在进行中，跳过启动")
                    return True  # 返回True表示学习状态正常
                
                # 设置该群组为学习状态
                self.learning_active[group_id] = True
                
                # 创建新的学习会话
                session_id = f"session_{group_id}_{int(time.time())}"
                self.current_session = LearningSession(
                    session_id=session_id,
                    start_time=datetime.now().isoformat()
                )
                # 保存新的学习会话到数据库
                await self.db_manager.save_learning_session_record(group_id, self.current_session.__dict__)
                
                logger.info(f"开始学习会话: {session_id} for group {group_id}")
                
                # 创建后台任务，确保不阻塞主线程
                learning_task = asyncio.create_task(self._learning_loop_safe(group_id))
                
                # 设置任务完成回调
                def on_learning_complete(task):
                    if task.exception():
                        logger.error(f"群组 {group_id} 学习任务异常完成: {task.exception()}")
                    else:
                        logger.info(f"群组 {group_id} 学习任务正常完成")
                    # 清除该群组的学习状态
                    self.learning_active[group_id] = False
                    
                learning_task.add_done_callback(on_learning_complete)
                
                return True
                
            except Exception as e:
                logger.error(f"启动群组 {group_id} 学习失败: {e}")
                # 确保清除学习状态
                self.learning_active[group_id] = False
                return False

    async def stop_learning(self, group_id: str = None):
        """停止学习流程"""
        if group_id:
            # 停止特定群组的学习
            self.learning_active[group_id] = False
            logger.info(f"停止群组 {group_id} 的学习任务")
        else:
            # 停止所有群组的学习
            for gid in list(self.learning_active.keys()):
                self.learning_active[gid] = False
            logger.info("停止所有群组的学习任务")
        
        if self.current_session:
            self.current_session.end_time = datetime.now().isoformat()
            self.current_session.success = True  # 假设正常停止即成功
            # 保存更新后的学习会话到数据库
            target_group_id = group_id or "global_learning"  # 使用指定的群组ID或默认值
            await self.db_manager.save_learning_session_record(target_group_id, self.current_session.__dict__)
            self.learning_sessions.append(self.current_session)  # 仍然添加到内存列表
            logger.info(f"学习会话结束: {self.current_session.session_id}")
            self.current_session = None

    async def _learning_loop_safe(self, group_id: str):
        """安全的学习循环 - 在后台线程执行，包含完整错误处理"""
        try:
            while self.learning_active.get(group_id, False):
                try:
                    # 检查是否应该暂停学习
                    should_pause, reason = await self.quality_monitor.should_pause_learning()
                    if should_pause:
                        logger.warning(f"群组 {group_id} 学习被暂停: {reason}")
                        await self.stop_learning(group_id)
                        break
                    
                    # 执行一个学习批次 - 在后台执行
                    await self._execute_learning_batch_background(group_id)
                    
                    # 等待下一个学习周期
                    await asyncio.sleep(self.learning_interval)
                    
                except asyncio.CancelledError:
                    logger.info(f"群组 {group_id} 学习任务被取消")
                    break
                except Exception as e:
                    logger.error(f"群组 {group_id} 学习循环异常: {e}", exc_info=True)
                    await asyncio.sleep(60)  # 异常时等待1分钟
        finally:
            # 确保清理资源
            if self.current_session:
                self.current_session.end_time = datetime.now().isoformat()
                await self.db_manager.save_learning_session_record(group_id, self.current_session.__dict__)
            logger.info(f"学习循环结束 for group {group_id}")

    async def _execute_learning_batch(self, group_id: str):
        """执行一个学习批次 - 集成强化学习"""
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
            
            # 3. 获取当前人格设置 (针对特定群组)
            current_persona = await self._get_current_persona(group_id)
            
            # 4. 【新增】强化学习记忆重放
            if self.config.enable_ml_analysis:
                try:
                    reinforcement_result = await self.ml_analyzer.reinforcement_memory_replay(
                        group_id, filtered_messages, current_persona
                    )
                    
                    if reinforcement_result and reinforcement_result.get('optimization_strategy'):
                        # 根据强化学习结果调整学习参数
                        learning_weight = reinforcement_result.get('optimization_strategy', {}).get('learning_weight', 1.0)
                        confidence_threshold = reinforcement_result.get('optimization_strategy', {}).get('confidence_threshold', self.config.confidence_threshold)
                        
                        # 动态调整筛选阈值
                        if confidence_threshold != self.config.confidence_threshold:
                            logger.info(f"根据强化学习调整置信度阈值: {self.config.confidence_threshold} -> {confidence_threshold}")
                            # 重新筛选消息（如果阈值提高了）
                            if confidence_threshold > self.config.confidence_threshold:
                                filtered_messages = [msg for msg in filtered_messages 
                                                   if msg.get('relevance_score', 0) >= confidence_threshold]
                                
                except Exception as e:
                    logger.error(f"强化学习记忆重放失败: {e}")
            
            # 5. 使用风格分析器深度分析
            style_analysis = await self.style_analyzer.analyze_conversation_style(group_id, filtered_messages)
            
            # 6. 【增强】使用提炼模型生成更新后的人格
            updated_persona = await self._generate_updated_persona_with_refinement(group_id, current_persona, style_analysis)
            
            # 7. 【新增】强化学习增量微调
            if self.config.enable_ml_analysis and updated_persona:
                try:
                    tuning_result = await self.ml_analyzer.reinforcement_incremental_tuning(
                        group_id, current_persona, updated_persona
                    )
                    
                    if tuning_result and tuning_result.get('updated_persona'):
                        # 使用强化学习优化后的人格
                        final_persona = tuning_result.get('updated_persona')
                        updated_persona.update(final_persona)
                        logger.info(f"应用强化学习优化后的人格，预期改进: {tuning_result.get('performance_prediction', {}).get('expected_improvement', 0)}")
                        
                except Exception as e:
                    logger.error(f"强化学习增量微调失败: {e}")
            
            # 8. 质量监控评估
            quality_metrics = await self.quality_monitor.evaluate_learning_batch(
                current_persona, 
                updated_persona, 
                filtered_messages
            )
            
            # 9. 根据质量评估决定是否应用更新
            if quality_metrics.consistency_score >= self.quality_threshold:
                await self._apply_learning_updates(group_id, style_analysis, filtered_messages)
                logger.info(f"学习更新已应用，质量得分: {quality_metrics.consistency_score:.3f} for group {group_id}")
                success = True
            else:
                logger.warning(f"学习质量不达标，跳过更新，得分: {quality_metrics.consistency_score:.3f} for group {group_id}")
                success = False
            
            # 10. 【新增】保存学习性能记录
            await self.db_manager.save_learning_performance_record(group_id, {
                'session_id': self.current_session.session_id if self.current_session else '',
                'timestamp': time.time(),
                'quality_score': quality_metrics.consistency_score,
                'learning_time': (datetime.now() - batch_start_time).total_seconds(),
                'success': success,
                'successful_pattern': json.dumps(style_analysis) if success else '',
                'failed_pattern': json.dumps({'reason': 'quality_threshold_not_met', 'score': quality_metrics.consistency_score}) if not success else ''
            })
            
            # 11. 标记消息为已处理
            await self._mark_messages_processed(unprocessed_messages)
            
            # 12. 更新学习会话统计并持久化
            if self.current_session:
                self.current_session.messages_processed += len(unprocessed_messages)
                self.current_session.filtered_messages += len(filtered_messages)
                self.current_session.quality_score = quality_metrics.consistency_score
                self.current_session.success = success
                # 每次批次结束都保存当前会话状态
                await self.db_manager.save_learning_session_record(group_id, self.current_session.__dict__)
            
            # 13. 【新增】定期执行策略优化
            if success and self.current_session and self.current_session.messages_processed % 500 == 0:
                try:
                    await self.ml_analyzer.reinforcement_strategy_optimization(group_id)
                    logger.info("执行了策略优化检查")
                except Exception as e:
                    logger.error(f"策略优化失败: {e}")
            
            # 记录批次耗时
            batch_duration = (datetime.now() - batch_start_time).total_seconds()
            logger.info(f"学习批次完成，耗时: {batch_duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"学习批次执行失败: {e}")
            raise LearningError(f"学习批次执行失败: {str(e)}")

    async def _execute_learning_batch_background(self, group_id: str):
        """在后台执行学习批次 - 使用线程池避免阻塞主协程"""
        try:
            batch_start_time = datetime.now()
            
            # 1. 异步获取数据
            unprocessed_messages = await self.message_collector.get_unprocessed_messages(
                limit=self.batch_size
            )
            
            if not unprocessed_messages:
                logger.debug("没有未处理的消息，跳过此批次")
                return
            
            logger.info(f"开始后台处理 {len(unprocessed_messages)} 条消息")
            
            # 2. 并行执行筛选和获取人格
            filtered_messages, current_persona = await asyncio.gather(
                self._filter_messages_with_context(unprocessed_messages),
                self._get_current_persona(group_id),
                return_exceptions=True
            )
            
            # 处理异常结果
            if isinstance(filtered_messages, Exception):
                logger.error(f"消息筛选异常: {filtered_messages}")
                filtered_messages = []
            
            if isinstance(current_persona, Exception):
                logger.error(f"获取人格异常: {current_persona}")
                current_persona = {}
            
            if not filtered_messages:
                logger.debug("没有通过筛选的消息")
                await self._mark_messages_processed(unprocessed_messages)
                return
            
            # 3. 并行执行强化学习和风格分析
            reinforcement_result, style_analysis = await asyncio.gather(
                self._execute_reinforcement_learning_background(group_id, filtered_messages, current_persona),
                self._execute_style_analysis_background(group_id, filtered_messages),
                return_exceptions=True
            )
            
            # 处理异常结果
            if isinstance(reinforcement_result, Exception):
                logger.error(f"强化学习异常: {reinforcement_result}")
                reinforcement_result = {}
            
            if isinstance(style_analysis, Exception):
                logger.error(f"风格分析异常: {style_analysis}")
                style_analysis = {}
            
            # 4. 动态调整学习参数（基于强化学习结果）
            if reinforcement_result and reinforcement_result.get('optimization_strategy'):
                confidence_threshold = reinforcement_result.get('optimization_strategy', {}).get('confidence_threshold', self.config.confidence_threshold)
                if confidence_threshold > self.config.confidence_threshold:
                    filtered_messages = [msg for msg in filtered_messages 
                                       if msg.get('relevance_score', 0) >= confidence_threshold]
                    logger.info(f"根据强化学习调整置信度阈值: {self.config.confidence_threshold} -> {confidence_threshold}")
            
            # 5. 使用提炼模型生成更新后的人格
            updated_persona = await self._generate_updated_persona_with_refinement(
                group_id, current_persona, style_analysis
            )
            
            # 6. 强化学习增量微调
            if self.config.enable_ml_analysis and updated_persona:
                tuning_result = await self._execute_incremental_tuning_background(
                    group_id, current_persona, updated_persona
                )
                if tuning_result and tuning_result.get('updated_persona'):
                    updated_persona.update(tuning_result.get('updated_persona'))
                    logger.info(f"应用强化学习优化，预期改进: {tuning_result.get('performance_prediction', {}).get('expected_improvement', 0)}")
            
            # 7. 质量评估和应用更新
            await self._finalize_learning_batch(
                group_id, current_persona, updated_persona, filtered_messages, 
                unprocessed_messages, batch_start_time
            )
            
        except Exception as e:
            logger.error(f"后台学习批次执行失败: {e}", exc_info=True)

    async def _execute_reinforcement_learning_background(self, group_id: str, filtered_messages, current_persona):
        """在后台执行强化学习"""
        if not self.config.enable_ml_analysis:
            return {}
        
        try:
            return await self.ml_analyzer.reinforcement_memory_replay(
                group_id, filtered_messages, current_persona
            )
        except Exception as e:
            logger.error(f"后台强化学习失败: {e}")
            return {}

    async def _execute_style_analysis_background(self, group_id: str, filtered_messages):
        """在后台执行风格分析"""
        try:
            return await self.style_analyzer.analyze_conversation_style(group_id, filtered_messages)
        except Exception as e:
            logger.error(f"后台风格分析失败: {e}")
            return {}

    async def _execute_incremental_tuning_background(self, group_id: str, base_persona, incremental_updates):
        """在后台执行增量微调"""
        try:
            return await self.ml_analyzer.reinforcement_incremental_tuning(
                group_id, base_persona, incremental_updates
            )
        except Exception as e:
            logger.error(f"后台增量微调失败: {e}")
            return {}

    async def _finalize_learning_batch(self, group_id: str, current_persona, updated_persona, 
                                     filtered_messages, unprocessed_messages, batch_start_time):
        """完成学习批次的最终处理"""
        try:
            # 质量监控评估
            quality_metrics = await self.quality_monitor.evaluate_learning_batch(
                current_persona, updated_persona, filtered_messages
            )
            
            # 根据质量评估决定是否应用更新
            success = False
            if quality_metrics.consistency_score >= self.quality_threshold:
                await self._apply_learning_updates(group_id, {}, filtered_messages)  # style_analysis may be empty
                logger.info(f"学习更新已应用，质量得分: {quality_metrics.consistency_score:.3f} for group {group_id}")
                success = True
            else:
                logger.warning(f"学习质量不达标，跳过更新，得分: {quality_metrics.consistency_score:.3f} for group {group_id}")
            
            # 保存学习性能记录
            await self.db_manager.save_learning_performance_record(group_id, {
                'session_id': self.current_session.session_id if self.current_session else '',
                'timestamp': time.time(),
                'quality_score': quality_metrics.consistency_score,
                'learning_time': (datetime.now() - batch_start_time).total_seconds(),
                'success': success,
                'successful_pattern': json.dumps({}) if success else '',
                'failed_pattern': json.dumps({'reason': 'quality_threshold_not_met', 'score': quality_metrics.consistency_score}) if not success else ''
            })
            
            # 标记消息为已处理
            await self._mark_messages_processed(unprocessed_messages)
            
            # 更新会话统计
            if self.current_session:
                self.current_session.messages_processed += len(unprocessed_messages)
                self.current_session.filtered_messages += len(filtered_messages)
                self.current_session.quality_score = quality_metrics.consistency_score
                self.current_session.success = success
                await self.db_manager.save_learning_session_record(group_id, self.current_session.__dict__)
            
            # 定期执行策略优化 - 不阻塞主流程
            if success and self.current_session and self.current_session.messages_processed % 500 == 0:
                asyncio.create_task(self._execute_strategy_optimization_background(group_id))
            
            batch_duration = (datetime.now() - batch_start_time).total_seconds()
            logger.info(f"后台学习批次完成，耗时: {batch_duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"完成学习批次失败: {e}")

    async def _execute_strategy_optimization_background(self, group_id: str):
        """在后台执行策略优化，不阻塞主流程"""
        try:
            await self.ml_analyzer.reinforcement_strategy_optimization(group_id)
            logger.info("后台策略优化完成")
        except Exception as e:
            logger.error(f"后台策略优化失败: {e}")

    async def _generate_updated_persona_with_refinement(self, group_id: str, current_persona: Dict[str, Any], style_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """使用提炼模型生成更新后的人格"""
        try:
            # 使用提炼模型生成人格更新
            from ..core.factory import LLMClient
            
            # 获取提炼模型客户端
            refine_client = None
            if hasattr(self, 'multidimensional_analyzer') and hasattr(self.multidimensional_analyzer, 'refine_llm_client'):
                refine_client = self.multidimensional_analyzer.refine_llm_client
            
            if not refine_client:
                logger.warning("提炼模型客户端未找到，使用传统方法生成人格")
                return await self._generate_updated_persona(group_id, current_persona, style_analysis)
            
            # 准备输入数据
            current_persona_json = json.dumps(current_persona, ensure_ascii=False, indent=2)
            style_analysis_json = json.dumps(style_analysis, ensure_ascii=False, indent=2)
            
            # 调用提炼模型
            response = await refine_client.chat_completion(
                prompt=self.prompts.PROGRESSIVE_LEARNING_GENERATE_UPDATED_PERSONA_PROMPT.format(
                    current_persona_json=current_persona_json,
                    style_analysis_json=style_analysis_json
                ),
                model_name=self.config.refine_model_name
            )
            
            if response and response.text():
                # 清理响应文本，移除markdown标识符
                clean_response = self._clean_llm_json_response(response.text())
                
                try:
                    updated_persona = safe_parse_llm_json(clean_response)
                    logger.info("使用提炼模型成功生成更新后的人格")
                    return updated_persona
                except json.JSONDecodeError as e:
                    logger.error(f"提炼模型返回的JSON格式不正确: {e}, 响应: {clean_response}")
                    return await self._generate_updated_persona(group_id, current_persona, style_analysis)
            
            return await self._generate_updated_persona(group_id, current_persona, style_analysis)
            
        except Exception as e:
            logger.error(f"使用提炼模型生成人格失败: {e}")
            return await self._generate_updated_persona(group_id, current_persona, style_analysis)

    def _clean_llm_json_response(self, response_text: str) -> str:
        """清理LLM响应中的markdown标识符和其他格式化字符"""
        import re
        
        # 移除markdown代码块标识符
        response_text = re.sub(r'```json\s*', '', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'```\s*$', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'^```\s*', '', response_text, flags=re.MULTILINE)
        
        # 移除其他常见的markdown标识符
        response_text = re.sub(r'^\s*```\w*\s*', '', response_text, flags=re.MULTILINE)
        
        # 寻找JSON对象的开始和结束
        # 找到第一个 { 和最后一个 }
        start = response_text.find('{')
        end = response_text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            response_text = response_text[start:end+1]
        
        # 清理多余的空白字符
        response_text = response_text.strip()
        
        return response_text

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
                # 使用专门的批量分析方法，不需要事件对象
                context_analysis = await self.multidimensional_analyzer.analyze_message_batch(
                    message['message'],
                    sender_id=message.get('sender_id', ''),
                    sender_name=message.get('sender_name', ''),
                    group_id=message.get('group_id', ''),
                    timestamp=message.get('timestamp', time.time())
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

    async def _get_current_persona(self, group_id: str) -> Dict[str, Any]:
        """获取当前人格设置 (针对特定群组)"""
        try:
            # 通过 PersonaManagerService 获取当前人格
            persona = await self.persona_manager.get_current_persona(group_id)
            if persona:
                return persona
            # 如果没有特定群组的人格，则返回默认结构
            return {
                'prompt': self.config.current_persona or "默认人格",
                'style_parameters': {},
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取当前人格失败 for group {group_id}: {e}")
            return {'prompt': '默认人格', 'style_parameters': {}}

    async def _generate_updated_persona(self, group_id: str, current_persona: Dict[str, Any], style_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成更新后的人格 - 直接在原有文本后面追加增量学习内容"""
        try:
            # 直接从框架获取当前人格
            provider = self.context.get_using_provider()
            if not provider or not provider.curr_personality:
                logger.warning(f"无法获取当前人格 for group {group_id}")
                return current_persona
            
            # 获取原有人格文本
            original_prompt = provider.curr_personality.get('prompt', '')
            
            # 构建增量学习内容
            learning_content = []
            
            if 'enhanced_prompt' in style_analysis:
                learning_content.append(style_analysis['enhanced_prompt'])
            
            if 'learning_insights' in style_analysis:
                insights = style_analysis['learning_insights']
                if insights:
                    learning_content.append(insights)
            
            # 直接在原有文本后面追加新内容
            if learning_content:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                new_content = f"\n\n【学习更新 - {timestamp}】\n" + "\n".join(learning_content)
                
                # 创建更新后的人格
                updated_persona = dict(provider.curr_personality)
                updated_persona['prompt'] = original_prompt + new_content
                updated_persona['last_updated'] = timestamp
                
                logger.info(f"直接追加学习内容到人格 for group {group_id}")
                return updated_persona
            else:
                logger.info(f"没有学习内容需要追加 for group {group_id}")
                return current_persona
                
        except Exception as e:
            logger.error(f"生成更新人格失败 for group {group_id}: {e}")
            return current_persona

    async def _apply_learning_updates(self, group_id: str, style_analysis: Dict[str, Any], messages: List[Dict[str, Any]]):
        """应用学习更新"""
        try:
            # 1. 更新人格prompt（通过 PersonaManagerService）
            logger.info(f"应用人格更新 for group {group_id}")
            update_success = await self.persona_manager.update_persona(group_id, style_analysis, messages)
            if not update_success:
                logger.error(f"通过 PersonaManagerService 更新人格失败 for group {group_id}")
            
            # 2. 记录学习更新
            if self.current_session:
                self.current_session.style_updates += 1
            
        except Exception as e:
            logger.error(f"应用学习更新失败 for group {group_id}: {e}")

    async def _mark_messages_processed(self, messages: List[Dict[str, Any]]):
        """标记消息为已处理"""
        message_ids = [msg['id'] for msg in messages if 'id' in msg]
        if message_ids:
            await self.message_collector.mark_messages_processed(message_ids)

    async def get_learning_status(self, group_id: str = None) -> Dict[str, Any]:
        """获取学习状态"""
        if group_id:
            # 获取特定群组的状态
            return {
                'learning_active': self.learning_active.get(group_id, False),
                'group_id': group_id,
                'current_session': self.current_session.__dict__ if self.current_session else None,
                'total_sessions': len(self.learning_sessions),
                'statistics': await self.message_collector.get_statistics(),
                'quality_report': await self.quality_monitor.get_quality_report(),
                'last_update': datetime.now().isoformat()
            }
        else:
            # 获取所有群组的状态
            return {
                'learning_active_groups': {gid: active for gid, active in self.learning_active.items()},
                'active_groups_count': sum(1 for active in self.learning_active.values() if active),
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

    async def stop(self):
        """停止服务"""
        try:
            await self.stop_learning()  # 停止所有群组的学习
            logger.info("渐进式学习服务已停止")
            return True
        except Exception as e:
            logger.error(f"停止渐进式学习服务失败: {e}")
            return False
