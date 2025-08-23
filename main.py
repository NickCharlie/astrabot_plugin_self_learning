"""
AstrBot 自学习插件 - 智能对话风格学习与人格优化
"""
import os
import json # 导入 json 模块
import asyncio
import time
import re # 导入正则表达式模块
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter
from astrbot.api.event.filter import PermissionType
import astrbot.api.star as star
from astrbot.api.star import register, Context
from astrbot.api import logger, AstrBotConfig
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .config import PluginConfig
from .core.factory import FactoryManager
from .exceptions import SelfLearningError
from .webui import Server, set_plugin_services # 导入 FastAPI 服务器相关
from .statics.messages import StatusMessages, CommandMessages, LogMessages, FileNames, DefaultValues

server_instance: Optional[Server] = None # 全局服务器实例
_server_cleanup_lock = asyncio.Lock() # 服务器清理锁，防止并发清理

@dataclass
class LearningStats:
    """学习统计信息"""
    total_messages_collected: int = 0
    filtered_messages: int = 0
    style_updates: int = 0
    persona_updates: int = 0
    last_learning_time: Optional[str] = None
    last_persona_update: Optional[str] = None


@register("astrbot_plugin_self_learning", "NickMo", "智能自学习对话插件", "1.2.1", "https://github.com/NickCharlie/astrbot_plugin_self_learning")
class SelfLearningPlugin(star.Star):
    """AstrBot 自学习插件 - 智能学习用户对话风格并优化人格设置"""

    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        
        # 初始化插件配置
        # 获取插件数据目录，并传递给 PluginConfig
        try:
            astrbot_data_path = get_astrbot_data_path()
            if astrbot_data_path is None:
                # 回退到当前目录下的 data 目录
                astrbot_data_path = os.path.join(os.path.dirname(__file__), "data")
                logger.warning("无法获取 AstrBot 数据路径，使用插件目录下的 data 目录")
            plugin_data_dir = os.path.join(astrbot_data_path, "plugins", "astrabot_plugin_self_learning")
            
            logger.info(f"插件数据目录: {plugin_data_dir}")
            self.plugin_config = PluginConfig.create_from_config(self.config, data_dir=plugin_data_dir)
            
        except Exception as e:
            logger.error(f"初始化插件配置失败: {e}")
            # 使用最保险的默认配置
            default_data_dir = os.path.join(os.path.dirname(__file__), "data")
            logger.warning(f"使用默认数据目录: {default_data_dir}")
            self.plugin_config = PluginConfig.create_from_config(self.config, data_dir=default_data_dir)
        
        # 确保数据目录存在
        os.makedirs(self.plugin_config.data_dir, exist_ok=True)
        
        # 初始化 messages_db_path 和 learning_log_path
        if not self.plugin_config.messages_db_path:
            self.plugin_config.messages_db_path = os.path.join(self.plugin_config.data_dir, FileNames.MESSAGES_DB_FILE)
        if not self.plugin_config.learning_log_path:
            self.plugin_config.learning_log_path = os.path.join(self.plugin_config.data_dir, FileNames.LEARNING_LOG_FILE)
        
        # 学习统计
        self.learning_stats = LearningStats()
        
        # 初始化服务层
        self._initialize_services()

        # 初始化 Web 服务器（但不启动，等待 on_load）
        global server_instance
        if self.plugin_config.enable_web_interface:
            server_instance = Server(port=self.plugin_config.web_interface_port)
            if server_instance:
                logger.info(StatusMessages.WEB_INTERFACE_ENABLED.format(host=server_instance.host, port=server_instance.port))
                logger.info("Web服务器实例已创建，将在on_load中启动")
            else:
                logger.error(StatusMessages.WEB_INTERFACE_INIT_FAILED)
        else:
            logger.info(StatusMessages.WEB_INTERFACE_DISABLED)
        
        logger.info(StatusMessages.PLUGIN_INITIALIZED)

    async def _start_web_server(self):
        """启动Web服务器的异步方法"""
        global server_instance
        if server_instance:
            logger.info(StatusMessages.WEB_SERVER_STARTING)
            try:
                await server_instance.start()
                logger.info(StatusMessages.WEB_SERVER_STARTED)
                
                # 启动数据库管理器
                await self.db_manager.start()
                logger.info(StatusMessages.DB_MANAGER_STARTED)
            except Exception as e:
                logger.error(StatusMessages.WEB_SERVER_START_FAILED.format(error=e), exc_info=True)

    def _initialize_services(self):
        """初始化所有服务层组件 - 使用工厂模式"""
        try:
            # 初始化工厂管理器
            self.factory_manager = FactoryManager()
            self.factory_manager.initialize_factories(self.plugin_config, self.context)
            
            # 获取服务工厂
            self.service_factory = self.factory_manager.get_service_factory()
            
            # 使用工厂创建核心服务
            self.db_manager = self.service_factory.create_database_manager()
            self.message_collector = self.service_factory.create_message_collector()
            self.multidimensional_analyzer = self.service_factory.create_multidimensional_analyzer()
            self.style_analyzer = self.service_factory.create_style_analyzer()
            self.quality_monitor = self.service_factory.create_quality_monitor()
            self.progressive_learning = self.service_factory.create_progressive_learning()
            self.intelligent_responder = self.service_factory.create_intelligent_responder()  # 重新启用智能回复器
            self.ml_analyzer = self.service_factory.create_ml_analyzer()
            self.persona_manager = self.service_factory.create_persona_manager()
            
            # 设置渐进式学习服务的增量更新回调函数，降低耦合性
            self.progressive_learning.set_update_system_prompt_callback(self._update_system_prompt_for_group)
            
            # 获取组件工厂并创建新的高级服务
            component_factory = self.factory_manager.get_component_factory()
            self.data_analytics = component_factory.create_data_analytics_service()
            self.advanced_learning = component_factory.create_advanced_learning_service()
            self.enhanced_interaction = component_factory.create_enhanced_interaction_service()
            self.intelligence_enhancement = component_factory.create_intelligence_enhancement_service()
            self.affection_manager = component_factory.create_affection_manager_service()
            
            # 创建临时人格更新器
            self.temporary_persona_updater = self.service_factory.create_temporary_persona_updater()
            
            # 初始化内部组件
            self._setup_internal_components()

            logger.info(StatusMessages.FACTORY_SERVICES_INIT_COMPLETE)
            
        except SelfLearningError as sle:
            logger.error(StatusMessages.SERVICES_INIT_FAILED.format(error=sle))
            raise # Re-raise as this is an expected initialization failure
        except (TypeError, ValueError) as e: # Catch common initialization errors
            logger.error(StatusMessages.CONFIG_TYPE_ERROR.format(error=e), exc_info=True)
            raise SelfLearningError(StatusMessages.INIT_FAILED_GENERIC.format(error=str(e))) from e
        except Exception as e: # Catch any other unexpected errors
            logger.error(StatusMessages.UNKNOWN_INIT_ERROR.format(error=e), exc_info=True)
            raise SelfLearningError(StatusMessages.INIT_FAILED_GENERIC.format(error=str(e))) from e
    
    def _setup_internal_components(self):
        """设置内部组件 - 使用工厂模式"""
        # 获取组件工厂
        self.component_factory = self.factory_manager.get_component_factory()

        # QQ号过滤器
        self.qq_filter = self.component_factory.create_qq_filter()
        
        # 消息过滤器
        self.message_filter = self.component_factory.create_message_filter(self.context)
        
        # 人格更新器
        # PersonaUpdater 的创建现在需要 backup_manager，它是一个服务，也应该通过 ServiceFactory 获取
        persona_backup_manager_instance = self.service_factory.create_persona_backup_manager()
        self.persona_updater = self.component_factory.create_persona_updater(self.context, persona_backup_manager_instance)
        
        # 学习调度器
        self.learning_scheduler = self.component_factory.create_learning_scheduler(self)
        
        # 异步任务管理 - 增强后台任务管理
        self.background_tasks = set()
        self.learning_tasks = {}  # 按group_id管理学习任务
        
        # 启动自动学习（如果启用）
        if self.plugin_config.enable_auto_learning:
            # 延迟启动，避免在初始化时启动大量任务
            asyncio.create_task(self._delayed_auto_start_learning())
    
    async def on_load(self):
        """插件加载时启动 Web 服务器和数据库管理器"""
        logger.info(StatusMessages.ON_LOAD_START)
        
        # 启动数据库管理器，确保数据库表被创建
        try:
            await self.db_manager.start()
            logger.info(StatusMessages.DB_MANAGER_STARTED)
        except Exception as e:
            logger.error(StatusMessages.DB_MANAGER_START_FAILED.format(error=e), exc_info=True)
        
        # 启动好感度管理服务（包含随机情绪初始化）
        if self.plugin_config.enable_affection_system:
            try:
                await self.affection_manager.start()
                logger.info("好感度管理服务启动成功")
            except Exception as e:
                logger.error(f"好感度管理服务启动失败: {e}", exc_info=True)
        
        # 设置Web服务器的插件服务实例和启动Web服务器
        global server_instance
        if self.plugin_config.enable_web_interface and server_instance:
            # 设置插件服务
            try:
                await set_plugin_services(
                    self.plugin_config,
                    self.factory_manager, # 传递 factory_manager
                    None  # 不再传递已弃用的 LLMClient
                )
                logger.info("Web服务器插件服务设置完成")
            except Exception as e:
                logger.error(f"设置Web服务器插件服务失败: {e}", exc_info=True)
            
            # 启动Web服务器
            logger.info(StatusMessages.WEB_SERVER_PREPARE.format(host=server_instance.host, port=server_instance.port))
            try:
                await server_instance.start()
                logger.info(StatusMessages.WEB_SERVER_STARTED)
            except Exception as e:
                logger.error(StatusMessages.WEB_SERVER_START_FAILED.format(error=e), exc_info=True)
        else:
            if not self.plugin_config.enable_web_interface:
                logger.info(StatusMessages.WEB_INTERFACE_DISABLED_SKIP)
            if not server_instance:
                logger.error(StatusMessages.SERVER_INSTANCE_NULL)
        
        logger.info(StatusMessages.PLUGIN_LOAD_COMPLETE)

    async def _delayed_start_learning(self, group_id: str):
        """延迟启动学习服务"""
        try:
            await asyncio.sleep(3)  # 等待初始化完成
            await self.service_factory.initialize_all_services() # 确保所有服务初始化完成
            # 启动针对特定 group_id 的渐进式学习
            await self.progressive_learning.start_learning(group_id)
            logger.info(StatusMessages.AUTO_LEARNING_SCHEDULER_STARTED.format(group_id=group_id))
        except Exception as e:
            logger.error(StatusMessages.LEARNING_SERVICE_START_FAILED.format(group_id=group_id, error=e))

    async def _priority_update_incremental_content(self, group_id: str, sender_id: str, message_text: str, event: AstrMessageEvent):
        """
        优先更新增量内容 - 每收到一条消息都会立即调用
        确保所有增量更新内容都能优先加入到system_prompt中
        """
        try:
            logger.info(f"开始优先更新增量内容: group_id={group_id}, sender_id={sender_id[:8]}")
            
            # 1. 立即进行消息的多维度分析（实时分析）
            if hasattr(self, 'multidimensional_analyzer') and self.multidimensional_analyzer:
                try:
                    # 立即分析当前消息的上下文
                    analysis_result = await self.multidimensional_analyzer.analyze_message_context(
                        event, message_text
                    )
                    if analysis_result:
                        logger.info(f"实时多维度分析完成，包含 {len(analysis_result)} 个维度")
                except Exception as e:
                    logger.error(f"实时多维度分析失败: {e}")
            
            # 2. 立即更新用户画像和社交关系
            if hasattr(self, 'affection_manager') and self.affection_manager:
                try:
                    # 立即更新好感度和社交关系
                    affection_result = await self.affection_manager.process_message_interaction(
                        group_id, sender_id, message_text
                    )
                    if affection_result and affection_result.get('success'):
                        logger.debug(f"实时好感度更新完成: {affection_result}")
                except Exception as e:
                    logger.error(f"实时好感度更新失败: {e}")
            
            # 3. 立即进行情绪和风格分析
            if hasattr(self, 'style_analyzer') and self.style_analyzer:
                try:
                    # 获取最近的消息进行风格分析
                    recent_messages = await self.db_manager.get_recent_filtered_messages(group_id, limit=5)
                    # 添加当前消息
                    current_message = {
                        'message': message_text,
                        'sender_id': sender_id,
                        'timestamp': time.time()
                    }
                    analysis_messages = recent_messages + [current_message]
                    
                    # 立即分析消息的风格
                    style_result = await self.style_analyzer.analyze_conversation_style(
                        group_id, analysis_messages
                    )
                    if style_result:
                        logger.debug(f"实时风格分析完成: {style_result}")
                except Exception as e:
                    logger.error(f"实时风格分析失败: {e}")
            
            # 4. 立即应用所有增量更新到system_prompt
            try:
                success = await self._update_system_prompt_for_group(group_id)
                if success:
                    logger.info(f"群组 {group_id} 增量更新优先应用到system_prompt成功")
                else:
                    logger.warning(f"群组 {group_id} 增量更新应用失败")
            except Exception as e:
                logger.error(f"增量更新应用异常 (群:{group_id}): {e}", exc_info=True)
            
            # 5. 如果启用实时学习，立即进行深度分析
            if self.plugin_config.enable_realtime_learning:
                try:
                    await self._process_message_realtime(group_id, message_text, sender_id)
                    logger.debug(f"实时学习处理完成: {group_id}")
                except Exception as e:
                    logger.error(f"实时学习处理失败: {e}")
            
            logger.info(f"增量内容优先更新流程完成: {group_id}")
            
        except Exception as e:
            logger.error(f"优先更新增量内容异常: {e}", exc_info=True)

    async def _update_system_prompt_for_group(self, group_id: str):
        """
        为特定群组实时更新system_prompt，集成所有可用的增量更新
        """
        try:
            # 收集当前群组的各种增量更新数据
            update_data = {}
            recent_messages = []  # 初始化变量
            
            # 1. 获取用户档案信息
            try:
                # 从多维分析器获取用户档案
                if hasattr(self, 'multidimensional_analyzer') and self.multidimensional_analyzer:
                    # 获取群组中最活跃的用户信息
                    user_profiles = getattr(self.multidimensional_analyzer, 'user_profiles', {})
                    if user_profiles:
                        # 合并所有用户的信息作为群组特征
                        communication_styles = []
                        activity_patterns = []
                        emotional_tendencies = []
                        
                        for user_id, profile in user_profiles.items():
                            if hasattr(profile, 'communication_style') and profile.communication_style:
                                # 转换沟通风格为可读描述
                                style_desc = self._format_communication_style(profile.communication_style)
                                if style_desc:
                                    communication_styles.append(style_desc)
                            if hasattr(profile, 'activity_pattern') and profile.activity_pattern:
                                activity_patterns.append(f"用户{user_id[:6]}活跃度{profile.activity_pattern.get('frequency', '普通')}")
                            if hasattr(profile, 'emotional_tendency') and profile.emotional_tendency:
                                # 转换情感倾向为可读描述
                                emotion_desc = self._format_emotional_tendency(profile.emotional_tendency)
                                if emotion_desc:
                                    emotional_tendencies.append(emotion_desc)
                        
                        if communication_styles or activity_patterns or emotional_tendencies:
                            update_data['user_profile'] = {
                                'preferences': '; '.join(activity_patterns[:3]) if activity_patterns else '',
                                'communication_style': '; '.join(communication_styles[:2]) if communication_styles else '',
                                'personality_traits': '; '.join(emotional_tendencies[:2]) if emotional_tendencies else ''
                            }
            except Exception as e:
                logger.debug(f"获取用户档案信息失败: {e}")
            
            # 2. 获取社交关系信息
            try:
                # 从数据库获取最近的群组互动信息
                recent_messages = await self.db_manager.get_recent_filtered_messages(group_id, limit=10)
                if recent_messages and len(recent_messages) > 1:
                    # 分析群组氛围
                    message_count = len(recent_messages)
                    unique_users = len(set(msg['sender_id'] for msg in recent_messages))
                    
                    if unique_users > 1:
                        atmosphere = f"活跃群聊，{unique_users}人参与"
                    else:
                        atmosphere = "私聊对话"
                        
                    update_data['social_relationship'] = {
                        'user_relationships': f"群组成员{unique_users}人",
                        'group_atmosphere': atmosphere,
                        'interaction_style': f"近期消息{message_count}条"
                    }
            except Exception as e:
                logger.debug(f"获取社交关系信息失败: {e}")
            
            # 3. 获取上下文感知信息
            try:
                # 从最近的消息中分析对话状态
                if recent_messages and len(recent_messages) > 0:
                    latest_msg = recent_messages[0]['message'] if recent_messages else ''
                    if latest_msg:
                        # 简单的话题提取（取前20个字符作为当前话题）
                        current_topic = latest_msg[:20] + '...' if len(latest_msg) > 20 else latest_msg
                        
                        update_data['context_awareness'] = {
                            'current_topic': current_topic,
                            'conversation_state': '进行中',
                            'dialogue_flow': f"最近{len(recent_messages)}条消息的对话"
                        }
            except Exception as e:
                logger.debug(f"获取上下文信息失败: {e}")
            
            # 4. 获取学习洞察信息
            try:
                # 从学习统计信息中获取基本洞察
                if hasattr(self, 'learning_stats') and self.learning_stats:
                    learning_info = {
                        'interaction_patterns': f"已学习消息: {getattr(self.learning_stats, 'total_messages_processed', 0)}条",
                        'improvement_suggestions': '基于历史对话的适应性调整',
                        'effective_strategies': '持续学习和优化中',
                        'learning_focus': '个性化交互改进'
                    }
                    
                    # 如果有处理过的消息，添加学习洞察
                    if getattr(self.learning_stats, 'total_messages_processed', 0) > 0:
                        update_data['learning_insights'] = learning_info
            except Exception as e:
                logger.debug(f"获取学习洞察失败: {e}")
            
            # 应用所有收集到的增量更新
            if update_data:
                success = await self.temporary_persona_updater.apply_comprehensive_update_to_system_prompt(
                    group_id, update_data
                )
                if success:
                    logger.info(f"群组 {group_id} system_prompt实时更新成功，包含 {len(update_data)} 种类型的增量更新")
                    return True
                else:
                    logger.warning(f"群组 {group_id} system_prompt更新失败")
                    return False
            else:
                logger.debug(f"群组 {group_id} 暂无可用的增量更新数据")
                return True  # 没有数据也算成功
                
        except Exception as e:
            logger.error(f"群组 {group_id} 实时更新system_prompt异常: {e}", exc_info=True)
            return False

    def _is_plugin_command(self, message_text: str) -> bool:
        """使用正则表达式检查消息是否为插件命令"""
        if not message_text:
            return False
        
        # 定义所有插件命令（不包含前缀符号）
        plugin_commands = [
            'learning_status',
            'start_learning', 
            'stop_learning',
            'force_learning',
            'clear_data',
            'export_data',
            'affection_status',
            'set_mood',
            'analytics_report',
            'persona_switch',
            'temp_persona',
            'apply_persona_updates',
            'clean_duplicate_content'
        ]
        
        # 创建命令的正则表达式模式
        # 匹配: [任意单个字符][命令名][可选的空格和参数]
        # ^.{1} : 开头任意一个字符（命令前缀）
        # (命令1|命令2|...) : 匹配任一插件命令
        # (\s.*)?$ : 可选的空格和参数，到字符串结尾
        commands_pattern = '|'.join(re.escape(cmd) for cmd in plugin_commands)
        pattern = rf'^.{{1}}({commands_pattern})(\s.*)?$'
        
        # 使用正则表达式匹配，忽略大小写
        return bool(re.match(pattern, message_text.strip(), re.IGNORECASE))

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息，收集用户对话数据"""
        
        # 检查是否启用消息抓取
        if not self.plugin_config.enable_message_capture:
            return
            
        try:
            group_id = event.get_group_id() or event.get_sender_id() # 使用群组ID或发送者ID作为会话ID
            sender_id = event.get_sender_id()
            
            # QQ号过滤
            if not self.qq_filter.should_collect_message(sender_id):
                return
                
            # 获取消息文本
            message_text = event.get_message_str()
            if not message_text or len(message_text.strip()) == 0:
                return
            
            # 过滤插件命令 - 避免命令被当作聊天消息处理
            if self._is_plugin_command(message_text):
                return
            
            # 优先更新增量内容 - 每收到消息都立即执行
            # 注释掉实时分析以提升回复速度，改为按配置定时分析
            # try:
            #     await self._priority_update_incremental_content(group_id, sender_id, message_text, event)
            #     logger.debug(f"优先增量内容更新完成: {group_id}")
            # except Exception as e:
            #     logger.error(f"优先增量内容更新失败: {e}")
                
            # 收集消息
            await self.message_collector.collect_message({
                'sender_id': sender_id,
                'sender_name': event.get_sender_name(),
                'message': message_text,
                'group_id': group_id,
                'timestamp': time.time(),
                'platform': event.get_platform_name()
            })
            
            self.learning_stats.total_messages_collected += 1
            
            # 处理好感度系统交互（如果启用）
            if self.plugin_config.enable_affection_system:
                try:
                    affection_result = await self.affection_manager.process_message_interaction(
                        group_id, sender_id, message_text
                    )
                    if affection_result.get('success'):
                        logger.debug(LogMessages.AFFECTION_PROCESSING_SUCCESS.format(result=affection_result))
                except Exception as e:
                    logger.error(LogMessages.AFFECTION_PROCESSING_FAILED.format(error=e))
            
            # 处理增强交互（多轮对话管理）
            try:
                await self.enhanced_interaction.update_conversation_context(
                    group_id, sender_id, message_text
                )
            except Exception as e:
                logger.error(LogMessages.ENHANCED_INTERACTION_FAILED.format(error=e))
            
            # 如果启用实时学习，立即进行筛选
            if self.plugin_config.enable_realtime_learning:
                await self._process_message_realtime(group_id, message_text, sender_id)
            
            # 智能启动学习任务（基于消息活动）
            await self._smart_start_learning_for_group(group_id)
            
            # 智能回复处理 - 在所有数据处理完成后
            try:
                intelligent_reply_params = await self.intelligent_responder.send_intelligent_response(event)
                if intelligent_reply_params:
                    # 使用yield发送智能回复
                    yield event.request_llm(
                        prompt=intelligent_reply_params['prompt'],
                        session_id=intelligent_reply_params['session_id'],
                        conversation=intelligent_reply_params['conversation']
                    )
                    logger.info(f"已发送智能回复请求: prompt长度={len(intelligent_reply_params['prompt'])}字符, session_id={intelligent_reply_params['session_id']}")
            except Exception as e:
                logger.error(f"智能回复处理失败: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(StatusMessages.MESSAGE_COLLECTION_ERROR.format(error=e), exc_info=True)

    async def _smart_start_learning_for_group(self, group_id: str):
        """智能启动群组学习任务 - 不阻塞主线程"""
        try:
            # 检查该群组是否已有学习任务
            if group_id in self.learning_tasks:
                return
            
            # 检查群组消息数量是否达到学习阈值
            stats = await self.message_collector.get_statistics(group_id)
            if stats.get('total_messages', 0) < self.plugin_config.min_messages_for_learning:
                return
            
            # 创建学习任务
            learning_task = asyncio.create_task(self._start_group_learning(group_id))
            
            # 设置完成回调
            def on_learning_task_complete(task):
                if group_id in self.learning_tasks:
                    del self.learning_tasks[group_id]
                if task.exception():
                    logger.error(f"群组 {group_id} 学习任务异常: {task.exception()}")
                else:
                    logger.info(f"群组 {group_id} 学习任务完成")
            
            learning_task.add_done_callback(on_learning_task_complete)
            self.learning_tasks[group_id] = learning_task
            
            logger.info(f"为群组 {group_id} 启动了智能学习任务")
            
        except Exception as e:
            logger.error(f"智能启动学习失败: {e}")

    async def _start_group_learning(self, group_id: str):
        """启动特定群组的学习任务"""
        try:
            success = await self.progressive_learning.start_learning(group_id)
            if success:
                logger.info(f"群组 {group_id} 学习任务启动成功")
            else:
                logger.warning(f"群组 {group_id} 学习任务启动失败")
        except Exception as e:
            logger.error(f"群组 {group_id} 学习任务启动异常: {e}")

    async def _delayed_auto_start_learning(self):
        """延迟自动启动学习 - 避免初始化时阻塞"""
        try:
            # 等待系统初始化完成
            await asyncio.sleep(30)
            
            # 获取活跃群组列表
            active_groups = await self._get_active_groups()
            
            for group_id in active_groups:
                try:
                    await self._smart_start_learning_for_group(group_id)
                    # 避免同时启动过多任务
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"延迟启动群组 {group_id} 学习失败: {e}")
                    
        except Exception as e:
            logger.error(f"延迟自动启动学习失败: {e}")

    async def _get_active_groups(self) -> List[str]:
        """获取活跃群组列表"""
        try:
            # 获取最近有消息的群组
            conn = await self.db_manager._get_messages_db_connection()
            cursor = await conn.cursor()
            
            # 获取最近24小时内有消息的群组
            cutoff_time = time.time() - 86400
            await cursor.execute('''
                SELECT DISTINCT group_id, COUNT(*) as msg_count
                FROM raw_messages 
                WHERE timestamp > ? AND group_id IS NOT NULL
                GROUP BY group_id
                HAVING msg_count >= ?
                ORDER BY msg_count DESC
                LIMIT 10
            ''', (cutoff_time, self.plugin_config.min_messages_for_learning))
            
            active_groups = []
            for row in await cursor.fetchall():
                if row[0]:  # 确保group_id不为空
                    active_groups.append(row[0])
                    
            logger.info(f"发现 {len(active_groups)} 个活跃群组")
            return active_groups
            
        except Exception as e:
            logger.error(f"获取活跃群组失败: {e}")
            return []

    async def _process_message_realtime(self, group_id: str, message_text: str, sender_id: str):
        """实时处理消息"""
        try:
            # 使用弱模型筛选消息
            current_persona_description = await self.persona_manager.get_current_persona_description()
            
            # 删除了智能回复相关处理
            # 原智能回复功能已移除
            
            if await self.multidimensional_analyzer.filter_message_with_llm(message_text, current_persona_description):
                await self.message_collector.add_filtered_message({
                    'message': message_text,
                    'sender_id': sender_id,
                    'group_id': group_id,
                    'timestamp': time.time(),
                    'confidence': 0.8  # 实时筛选置信度
                })
                self.learning_stats.filtered_messages += 1
                
        except Exception as e:
            logger.error(StatusMessages.REALTIME_PROCESSING_ERROR.format(error=e), exc_info=True)

    @filter.command("learning_status")
    @filter.permission_type(PermissionType.ADMIN)
    async def learning_status_command(self, event: AstrMessageEvent):
        """查看学习状态"""
        try:
            group_id = event.get_group_id() or event.get_sender_id() # 获取当前会话ID
            
            # 获取收集统计
            collector_stats = await self.message_collector.get_statistics(group_id) # 传入 group_id
            
            # 确保 collector_stats 不为 None
            if collector_stats is None:
                collector_stats = {
                    'total_messages': 0,
                    'filtered_messages': 0,
                    'raw_messages': 0,
                    'unprocessed_messages': 0,
                }
            
            # 获取当前人格设置
            current_persona_info = await self.persona_manager.get_current_persona(group_id)
            current_persona_name = CommandMessages.STATUS_UNKNOWN
            if current_persona_info and isinstance(current_persona_info, dict):
                current_persona_name = current_persona_info.get('name', CommandMessages.STATUS_UNKNOWN)
            
            # 获取渐进式学习服务的状态
            learning_status = await self.progressive_learning.get_learning_status()
            
            # 确保 learning_status 不为 None
            if learning_status is None:
                learning_status = {
                    'learning_active': False,
                    'current_session': None,
                    'total_sessions': 0,
                }
            
            # 构建状态信息
            status_info = CommandMessages.STATUS_REPORT_HEADER.format(group_id=group_id)
            
            # 基础配置
            status_info += CommandMessages.STATUS_BASIC_CONFIG.format(
                message_capture=CommandMessages.STATUS_ENABLED if self.plugin_config.enable_message_capture else CommandMessages.STATUS_DISABLED,
                auto_learning=CommandMessages.STATUS_ENABLED if self.plugin_config.enable_auto_learning else CommandMessages.STATUS_DISABLED,
                realtime_learning=CommandMessages.STATUS_ENABLED if self.plugin_config.enable_realtime_learning else CommandMessages.STATUS_DISABLED,
                web_interface=CommandMessages.STATUS_ENABLED if self.plugin_config.enable_web_interface else CommandMessages.STATUS_DISABLED
            )
            
            # 抓取设置
            status_info += CommandMessages.STATUS_CAPTURE_SETTINGS.format(
                target_qq=self.plugin_config.target_qq_list if self.plugin_config.target_qq_list else CommandMessages.STATUS_ALL_USERS,
                current_persona=current_persona_name
            )
            
            # Provider配置信息
            if hasattr(self, 'llm_adapter') and self.llm_adapter:
                provider_info = self.llm_adapter.get_provider_info()
                status_info += CommandMessages.STATUS_MODEL_CONFIG.format(
                    filter_model=provider_info.get('filter', '未配置'),
                    refine_model=provider_info.get('refine', '未配置')
                )
            else:
                status_info += CommandMessages.STATUS_MODEL_CONFIG.format(
                    filter_model='未配置框架Provider',
                    refine_model='未配置框架Provider'
                )
            
            # 学习统计
            status_info += CommandMessages.STATUS_LEARNING_STATS.format(
                total_messages=collector_stats.get('total_messages', 0),
                filtered_messages=collector_stats.get('filtered_messages', 0),
                style_updates=learning_status.get('current_session', {}).get('style_updates', 0),
                last_learning_time=learning_status.get('current_session', {}).get('end_time', CommandMessages.STATUS_NEVER_EXECUTED)
            )
            
            # 存储统计
            status_info += CommandMessages.STATUS_STORAGE_STATS.format(
                raw_messages=collector_stats.get('raw_messages', 0),
                unprocessed_messages=collector_stats.get('unprocessed_messages', 0),
                filtered_messages=collector_stats.get('filtered_messages', 0)
            )
            
            # 调度状态
            scheduler_status = CommandMessages.STATUS_RUNNING if learning_status.get('learning_active') else CommandMessages.STATUS_STOPPED
            status_info += "\n\n" + CommandMessages.STATUS_SCHEDULER.format(status=scheduler_status)

            yield event.plain_result(status_info.strip())
            
        except Exception as e:
            logger.error(CommandMessages.ERROR_GET_LEARNING_STATUS.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.STATUS_QUERY_FAILED.format(error=str(e)))

    @filter.command("start_learning")
    @filter.permission_type(PermissionType.ADMIN)
    async def start_learning_command(self, event: AstrMessageEvent):
        """手动启动学习"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            
            if await self.progressive_learning.start_learning(group_id):
                yield event.plain_result(CommandMessages.LEARNING_STARTED.format(group_id=group_id))
            else:
                yield event.plain_result(CommandMessages.LEARNING_RUNNING.format(group_id=group_id))
            
        except Exception as e:
            logger.error(CommandMessages.ERROR_START_LEARNING.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.STARTUP_FAILED.format(error=str(e)))

    @filter.command("stop_learning")
    @filter.permission_type(PermissionType.ADMIN)
    async def stop_learning_command(self, event: AstrMessageEvent):
        """停止学习"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            
            # ProgressiveLearningService 的 stop_learning 目前没有 group_id 参数
            # 如果需要停止特定 group_id 的学习，ProgressiveLearningService 需要修改
            # 暂时调用全局停止，或者假设 stop_learning 会停止当前活跃的会话
            await self.progressive_learning.stop_learning()
            yield event.plain_result(CommandMessages.LEARNING_STOPPED.format(group_id=group_id))
            
        except Exception as e:
            logger.error(CommandMessages.ERROR_STOP_LEARNING.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.STOP_FAILED.format(error=str(e)))

    @filter.command("force_learning")  
    @filter.permission_type(PermissionType.ADMIN)
    async def force_learning_command(self, event: AstrMessageEvent):
        """强制执行一次学习周期"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            yield event.plain_result(CommandMessages.FORCE_LEARNING_START.format(group_id=group_id))
            
            # 直接调用 ProgressiveLearningService 的批处理方法
            await self.progressive_learning._execute_learning_batch(group_id)
            
            yield event.plain_result(CommandMessages.FORCE_LEARNING_COMPLETE.format(group_id=group_id))
            
        except Exception as e:
            logger.error(CommandMessages.ERROR_FORCE_LEARNING.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.ERROR_FORCE_LEARNING.format(error=str(e)))

    @filter.command("clear_data")
    @filter.permission_type(PermissionType.ADMIN)
    async def clear_data_command(self, event: AstrMessageEvent):
        """清空学习数据"""
        try:
            await self.message_collector.clear_all_data()
            
            # 重置统计
            self.learning_stats = LearningStats()
            
            yield event.plain_result(CommandMessages.DATA_CLEARED)
            
        except Exception as e: # Consider more specific exceptions if possible
            logger.error(CommandMessages.ERROR_CLEAR_DATA.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.ERROR_CLEAR_DATA.format(error=str(e)))

    @filter.command("export_data")
    @filter.permission_type(PermissionType.ADMIN)
    async def export_data_command(self, event: AstrMessageEvent):
        """导出学习数据"""
        try:
            export_data = await self.message_collector.export_learning_data()
            
            # 生成导出文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = FileNames.EXPORT_FILENAME_TEMPLATE.format(timestamp=timestamp)
            filepath = os.path.join(self.plugin_config.data_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
                
            yield event.plain_result(CommandMessages.DATA_EXPORTED.format(filepath=filepath))
            
        except Exception as e: # Consider more specific exceptions if possible
            logger.error(CommandMessages.ERROR_EXPORT_DATA.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.ERROR_EXPORT_DATA.format(error=str(e)))

    @filter.command("affection_status")
    @filter.permission_type(PermissionType.ADMIN)
    async def affection_status_command(self, event: AstrMessageEvent):
        """查看好感度状态"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            user_id = event.get_sender_id()
            
            if not self.plugin_config.enable_affection_system:
                yield event.plain_result(CommandMessages.AFFECTION_DISABLED)
                return
                
            # 获取好感度状态
            affection_status = await self.affection_manager.get_affection_status(group_id)
            
            # 确保当前群组有情绪状态（如果没有会自动创建随机情绪）
            current_mood = None
            if self.plugin_config.enable_startup_random_mood:
                current_mood = await self.affection_manager.ensure_mood_for_group(group_id)
            else:
                current_mood = await self.affection_manager.get_current_mood(group_id)
            
            # 获取用户个人好感度
            user_affection = await self.db_manager.get_user_affection(group_id, user_id)
            user_level = user_affection['affection_level'] if user_affection else 0
            
            status_info = CommandMessages.AFFECTION_STATUS_HEADER.format(group_id=group_id)
            status_info += "\n\n" + CommandMessages.AFFECTION_USER_LEVEL.format(
                user_level=user_level, max_affection=self.plugin_config.max_user_affection
            )
            status_info += "\n" + CommandMessages.AFFECTION_TOTAL_STATUS.format(
                total_affection=affection_status['total_affection'],
                max_total_affection=affection_status['max_total_affection']
            )
            status_info += "\n" + CommandMessages.AFFECTION_USER_COUNT.format(user_count=affection_status['user_count'])
            status_info += "\n\n" + CommandMessages.AFFECTION_CURRENT_MOOD
            
            if current_mood:
                mood_info = current_mood
                status_info += "\n" + CommandMessages.AFFECTION_MOOD_TYPE.format(mood_type=mood_info.mood_type.value)
                status_info += "\n" + CommandMessages.AFFECTION_MOOD_INTENSITY.format(intensity=mood_info.intensity)
                status_info += "\n" + CommandMessages.AFFECTION_MOOD_DESCRIPTION.format(description=mood_info.description)
            else:
                status_info += "\n" + CommandMessages.AFFECTION_NO_MOOD
                
            if affection_status['top_users']:
                status_info += "\n\n" + CommandMessages.AFFECTION_TOP_USERS
                for i, user in enumerate(affection_status['top_users'][:3], 1):
                    status_info += "\n" + CommandMessages.AFFECTION_USER_RANK.format(
                        rank=i, user_id=user['user_id'], affection_level=user['affection_level']
                    )
            
            yield event.plain_result(status_info)
            
        except Exception as e:
            logger.error(CommandMessages.ERROR_GET_AFFECTION_STATUS.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.ERROR_GET_AFFECTION_STATUS.format(error=str(e)))

    @filter.command("set_mood")
    @filter.permission_type(PermissionType.ADMIN)
    async def set_mood_command(self, event: AstrMessageEvent):
        """手动设置bot情绪（通过增量人格更新）"""
        try:
            if not self.plugin_config.enable_affection_system:
                yield event.plain_result(CommandMessages.AFFECTION_DISABLED)
                return
                
            args = event.get_message_str().split()[1:]  # 获取命令参数
            if len(args) < 1:
                yield event.plain_result("使用方法：/set_mood <mood_type>\n可用情绪: happy, sad, excited, calm, angry, anxious, playful, serious, nostalgic, curious")
                return
                
            group_id = event.get_group_id() or event.get_sender_id()
            mood_type = args[0].lower()
            
            # 验证情绪类型
            valid_moods = {
                'happy': '心情很好，说话比较活泼开朗，容易表达正面情感',
                'sad': '心情有些低落，说话比较温和，需要更多的理解和安慰',
                'excited': '很兴奋，说话比较有活力，对很多事情都很感兴趣',
                'calm': '心情平静，说话比较稳重，给人安全感',
                'angry': '心情不太好，说话可能比较直接，不太有耐心',
                'anxious': '有些紧张不安，说话可能比较谨慎，需要更多确认',
                'playful': '心情很调皮，喜欢开玩笑，说话比较幽默风趣',
                'serious': '比较严肃认真，说话简洁直接，专注于重要的事情',
                'nostalgic': '有些怀旧情绪，说话带有回忆色彩，比较感性',
                'curious': '对很多事情都很好奇，喜欢提问和探索新事物'
            }
            
            if mood_type not in valid_moods:
                yield event.plain_result(f"❌ 无效的情绪类型。支持的情绪: {', '.join(valid_moods.keys())}")
                return
            
            # 通过增量更新的方式设置情绪
            mood_description = valid_moods[mood_type]
            
            # 统一使用apply_mood_based_persona_update方法，它会同时处理文件和prompt更新
            persona_success = await self.temporary_persona_updater.apply_mood_based_persona_update(
                group_id, mood_type, mood_description
            )
            
            # 同时在affection_manager中记录情绪状态（但不重复添加到prompt）
            from .services.affection_manager import MoodType
            try:
                mood_enum = MoodType(mood_type)
                # 只记录到affection_manager的数据库，不更新prompt（避免重复）
                await self.affection_manager.db_manager.save_bot_mood(
                    group_id, mood_type, 0.7, mood_description, 
                    self.plugin_config.mood_persistence_hours or 24
                )
                # 更新内存缓存
                from .services.affection_manager import BotMood
                import time
                mood_obj = BotMood(
                    mood_type=mood_enum,
                    intensity=0.7,
                    description=mood_description,
                    start_time=time.time(),
                    duration_hours=self.plugin_config.mood_persistence_hours or 24
                )
                self.affection_manager.current_moods[group_id] = mood_obj
                affection_success = True
            except Exception as e:
                logger.warning(f"设置affection_manager情绪失败: {e}")
                affection_success = False
            
            if persona_success:
                status_msg = f"✅ 情绪状态已设置为: {mood_type}\n描述: {mood_description}"
                if not affection_success:
                    status_msg += "\n⚠️ 注意：情绪状态可能无法在状态查询中正确显示"
                yield event.plain_result(status_msg)
            else:
                yield event.plain_result(f"❌ 设置情绪状态失败")
            
        except Exception as e:
            logger.error(CommandMessages.ERROR_SET_MOOD.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.ERROR_SET_MOOD.format(error=str(e)))

    @filter.command("analytics_report")
    @filter.permission_type(PermissionType.ADMIN)
    async def analytics_report_command(self, event: AstrMessageEvent):
        """生成数据分析报告"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            
            yield event.plain_result(CommandMessages.ANALYTICS_GENERATING)
            
            # 生成学习轨迹图表
            chart_data = await self.data_analytics.generate_learning_trajectory_chart(group_id)
            
            # 生成用户行为分析
            behavior_analysis = await self.data_analytics.analyze_user_behavior_patterns(group_id)
            
            report_info = CommandMessages.ANALYTICS_REPORT_HEADER.format(group_id=group_id)
            
            report_info += CommandMessages.ANALYTICS_LEARNING_STATS.format(
                total_messages=chart_data.get('total_messages', 0),
                learning_sessions=chart_data.get('learning_sessions', 0),
                avg_quality=chart_data.get('avg_quality', 0)
            )
            
            report_info += CommandMessages.ANALYTICS_USER_BEHAVIOR.format(
                active_users=len(behavior_analysis.get('user_patterns', {})),
                main_topics=', '.join(behavior_analysis.get('common_topics', [])[:3]),
                emotion_tendency=behavior_analysis.get('dominant_emotion', '中性')
            )
            
            report_info += "\n\n" + CommandMessages.ANALYTICS_RECOMMENDATIONS.format(
                recommendations=behavior_analysis.get('recommendations', '继续保持当前学习模式')
            )
            
            yield event.plain_result(report_info)
            
        except Exception as e:
            logger.error(CommandMessages.ERROR_ANALYTICS_REPORT.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.ERROR_ANALYTICS_REPORT.format(error=str(e)))

    @filter.command("persona_switch")
    async def persona_switch_command(self, event: AstrMessageEvent):
        """切换人格模式"""
        try:
            args = event.get_message_str().split()[1:]  # 获取命令参数
            if len(args) < 1:
                yield event.plain_result(CommandMessages.PERSONA_SWITCH_USAGE)
                return
                
            group_id = event.get_group_id() or event.get_sender_id()
            persona_name = args[0]
            
            # 执行人格切换
            success = await self.advanced_learning.switch_persona(group_id, persona_name)
            
            if success:
                yield event.plain_result(CommandMessages.PERSONA_SWITCH_SUCCESS.format(persona_name=persona_name))
            else:
                yield event.plain_result(CommandMessages.PERSONA_SWITCH_FAILED)
                
        except Exception as e:
            logger.error(CommandMessages.ERROR_PERSONA_SWITCH.format(error=e), exc_info=True)
            yield event.plain_result(CommandMessages.ERROR_PERSONA_SWITCH.format(error=str(e)))

    @filter.command("temp_persona")
    @filter.permission_type(PermissionType.ADMIN)
    async def temp_persona_command(self, event: AstrMessageEvent):
        """临时人格更新命令"""
        try:
            args = event.get_message_str().split()
            if len(args) < 2:
                yield event.plain_result("使用方法：/temp_persona <操作> [参数]\n操作：apply, status, remove, extend, backup_list, restore")
                return
            
            operation = args[1].lower()
            group_id = event.get_group_id() or event.get_sender_id()
            
            if operation == "apply":
                # 应用临时人格: /temp_persona apply "特征1,特征2" "对话1|对话2" [持续时间分钟]
                if len(args) < 4:
                    yield event.plain_result("使用方法：/temp_persona apply \"特征1,特征2\" \"对话1|对话2\" [持续时间分钟]")
                    return
                
                features_str = args[2].strip('"')
                dialogs_str = args[3].strip('"')
                duration = int(args[4]) if len(args) > 4 else 60
                
                features = [f.strip() for f in features_str.split(',') if f.strip()]
                dialogs = [d.strip() for d in dialogs_str.split('|') if d.strip()]
                
                success = await self.temporary_persona_updater.apply_temporary_persona_update(
                    group_id, features, dialogs, duration
                )
                
                if success:
                    yield event.plain_result(f"✅ 临时人格已应用，持续时间: {duration}分钟\n特征数量: {len(features)}\n对话数量: {len(dialogs)}")
                else:
                    yield event.plain_result("❌ 临时人格应用失败")
            
            elif operation == "status":
                # 查看临时人格状态
                status = await self.temporary_persona_updater.get_temporary_persona_status(group_id)
                if status:
                    remaining_minutes = status['remaining_seconds'] // 60
                    yield event.plain_result(f"""📊 临时人格状态:
                        人格名称: {status['persona_name']}
                        剩余时间: {remaining_minutes}分钟
                        特征数量: {status['features_count']}
                        对话数量: {status['dialogs_count']}
                        备份文件: {os.path.basename(status['backup_path'])}""")
                else:
                    yield event.plain_result("ℹ️ 当前没有活动的临时人格")
            
            elif operation == "remove":
                # 移除临时人格
                success = await self.temporary_persona_updater.remove_temporary_persona(group_id)
                if success:
                    yield event.plain_result("✅ 临时人格已移除，已恢复原始人格")
                else:
                    yield event.plain_result("ℹ️ 当前没有需要移除的临时人格")
            
            elif operation == "extend":
                # 延长临时人格: /temp_persona extend [分钟数]
                additional_minutes = int(args[2]) if len(args) > 2 else 30
                success = await self.temporary_persona_updater.extend_temporary_persona(group_id, additional_minutes)
                if success:
                    yield event.plain_result(f"✅ 临时人格时间已延长 {additional_minutes} 分钟")
                else:
                    yield event.plain_result("❌ 延长临时人格失败，可能没有活动的临时人格")
            
            elif operation == "backup_list":
                # 列出备份文件
                backups = await self.temporary_persona_updater.list_persona_backups(group_id)
                if backups:
                    backup_info = "📋 人格备份文件列表:\n"
                    for i, backup in enumerate(backups[:10], 1):  # 只显示前10个
                        backup_info += f"{i}. {backup['filename']}\n"
                        backup_info += f"   人格: {backup['persona_name']}\n"
                        backup_info += f"   时间: {backup['backup_time'][:16]}\n"
                        backup_info += f"   原因: {backup['backup_reason']}\n\n"
                    yield event.plain_result(backup_info.strip())
                else:
                    yield event.plain_result("ℹ️ 没有找到备份文件")
            
            elif operation == "restore":
                # 从备份恢复: /temp_persona restore [备份文件名]
                if len(args) < 3:
                    yield event.plain_result("请指定要恢复的备份文件名")
                    return
                
                backup_filename = args[2]
                backups = await self.temporary_persona_updater.list_persona_backups(group_id)
                
                target_backup = None
                for backup in backups:
                    if backup['filename'] == backup_filename:
                        target_backup = backup
                        break
                
                if target_backup:
                    success = await self.temporary_persona_updater.restore_from_backup_file(
                        group_id, target_backup['file_path']
                    )
                    if success:
                        yield event.plain_result(f"✅ 人格已从备份恢复: {backup_filename}")
                    else:
                        yield event.plain_result(f"❌ 从备份恢复失败: {backup_filename}")
                else:
                    yield event.plain_result(f"❌ 找不到备份文件: {backup_filename}")
            
            else:
                yield event.plain_result("❌ 无效的操作。支持的操作: apply, status, remove, extend, backup_list, restore")
                
        except Exception as e:
            logger.error(f"临时人格命令执行失败: {e}", exc_info=True)
            yield event.plain_result(f"临时人格命令执行失败: {str(e)}")


    @filter.command("apply_persona_updates")
    @filter.permission_type(PermissionType.ADMIN)
    async def apply_persona_updates_command(self, event: AstrMessageEvent):
        """应用persona_updates.txt中的增量人格更新"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            
            yield event.plain_result("🔄 开始应用增量人格更新...")
            
            # 调用临时人格更新器的方法
            success = await self.temporary_persona_updater.read_and_apply_persona_updates(group_id)
            
            if success:
                yield event.plain_result("✅ 增量人格更新应用成功！更新文件已清空，等待下次更新。")
            else:
                yield event.plain_result("ℹ️ 没有找到有效的人格更新内容，或更新应用失败。")
                
        except Exception as e:
            logger.error(f"应用人格更新命令失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 应用人格更新失败: {str(e)}")

    @filter.command("clean_duplicate_content")
    @filter.permission_type(PermissionType.ADMIN)
    async def clean_duplicate_content_command(self, event: AstrMessageEvent):
        """清理历史重复的情绪状态和增量更新内容"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            
            yield event.plain_result("🧹 开始清理重复的历史内容...")
            
            # 获取provider
            provider = self.context.get_using_provider()
            if not provider or not hasattr(provider, 'curr_personality') or not provider.curr_personality:
                yield event.plain_result("❌ 无法获取当前人格信息")
                return
            
            # 获取当前prompt
            current_prompt = provider.curr_personality.get('prompt', '')
            if not current_prompt:
                yield event.plain_result("ℹ️ 当前人格没有prompt内容")
                return
            
            # 记录清理前的长度
            original_length = len(current_prompt)
            
            # 使用清理函数
            cleaned_prompt = self.temporary_persona_updater._clean_duplicate_content(current_prompt)
            
            # 更新prompt
            provider.curr_personality['prompt'] = cleaned_prompt
            
            # 计算清理效果
            cleaned_length = len(cleaned_prompt)
            saved_chars = original_length - cleaned_length
            
            # 同时清理persona_updates.txt文件
            await self.temporary_persona_updater.clear_persona_updates_file()
            
            yield event.plain_result(f"✅ 重复内容清理完成！\n"
                                   f"📊 清理前长度: {original_length} 字符\n"
                                   f"📊 清理后长度: {cleaned_length} 字符\n"
                                   f"🗑️ 清理了 {saved_chars} 个重复字符\n"
                                   f"🧹 同时清空了persona_updates.txt文件")
                
        except Exception as e:
            logger.error(f"清理重复内容命令失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 清理重复内容失败: {str(e)}")

    async def terminate(self):
        """插件卸载时的清理工作 - 增强后台任务管理"""
        try:
            logger.info("开始插件清理工作...")
            
            # 1. 停止所有学习任务
            logger.info("停止所有学习任务...")
            for group_id, task in list(self.learning_tasks.items()):
                try:
                    # 先停止学习流程
                    await self.progressive_learning.stop_learning()
                    
                    # 取消学习任务
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    logger.info(f"群组 {group_id} 学习任务已停止")
                except Exception as e:
                    logger.error(f"停止群组 {group_id} 学习任务失败: {e}")
            
            self.learning_tasks.clear()
            
            # 2. 停止学习调度器
            if hasattr(self, 'learning_scheduler'):
                try:
                    await self.learning_scheduler.stop()
                    logger.info("学习调度器已停止")
                except Exception as e:
                    logger.error(f"停止学习调度器失败: {e}")
                
            # 3. 取消所有后台任务
            logger.info("取消所有后台任务...")
            for task in list(self.background_tasks):
                try:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                except Exception as e:
                    logger.error(LogMessages.BACKGROUND_TASK_CANCEL_ERROR.format(error=e))
            
            self.background_tasks.clear()
            
            # 4. 停止所有服务
            logger.info("停止所有服务...")
            if hasattr(self, 'factory_manager'):
                try:
                    await self.factory_manager.cleanup()
                    logger.info("服务工厂已清理")
                except Exception as e:
                    logger.error(f"清理服务工厂失败: {e}")
            
            # 5. 清理临时人格
            if hasattr(self, 'temporary_persona_updater'):
                try:
                    await self.temporary_persona_updater.cleanup_temp_personas()
                    logger.info("临时人格已清理")
                except Exception as e:
                    logger.error(f"清理临时人格失败: {e}")
                
            # 6. 保存最终状态
            if hasattr(self, 'message_collector'):
                try:
                    await self.message_collector.save_state()
                    logger.info("消息收集器状态已保存")
                except Exception as e:
                    logger.error(f"保存消息收集器状态失败: {e}")
                
            # 7. 停止 Web 服务器 - 增强版
            global server_instance, _server_cleanup_lock
            async with _server_cleanup_lock:
                if server_instance:
                    try:
                        logger.info(f"正在停止Web服务器 (端口: {server_instance.port})...")
                        
                        # 记录服务器信息用于日志
                        port = server_instance.port
                        
                        # 调用增强的停止方法
                        await server_instance.stop()
                        
                        # 额外等待确保端口释放
                        await asyncio.sleep(1)
                        
                        # 重置全局实例
                        server_instance = None
                        
                        logger.info(f"Web服务器已停止，端口 {port} 已释放")
                    except Exception as e:
                        logger.error(f"停止Web服务器失败: {e}", exc_info=True)
                        # 即使出错也要重置实例，避免重复尝试
                        server_instance = None
                else:
                    logger.info("Web服务器已经停止或未初始化")
                
            # 8. 保存配置到文件
            try:
                config_path = os.path.join(self.plugin_config.data_dir, 'config.json')
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.plugin_config.to_dict(), f, ensure_ascii=False, indent=2)
                logger.info(LogMessages.PLUGIN_CONFIG_SAVED)
            except Exception as e:
                logger.error(f"保存配置失败: {e}")
            
            logger.info(LogMessages.PLUGIN_UNLOAD_SUCCESS)
            
        except Exception as e:
            logger.error(LogMessages.PLUGIN_UNLOAD_CLEANUP_FAILED.format(error=e), exc_info=True)
    
    def _format_communication_style(self, communication_style: dict) -> str:
        """
        将沟通风格字典转换为可读描述
        
        Args:
            communication_style: 沟通风格字典
            
        Returns:
            str: 可读的描述文本
        """
        try:
            if not communication_style or not isinstance(communication_style, dict):
                return ""
            
            descriptions = []
            
            # 解析各种沟通风格特征
            if 'formality' in communication_style:
                formality = communication_style['formality']
                if formality > 0.7:
                    descriptions.append("正式礼貌")
                elif formality < 0.3:
                    descriptions.append("随意轻松")
                else:
                    descriptions.append("适中得体")
            
            if 'enthusiasm' in communication_style:
                enthusiasm = communication_style['enthusiasm']
                if enthusiasm > 0.7:
                    descriptions.append("热情活跃")
                elif enthusiasm < 0.3:
                    descriptions.append("冷静内敛")
            
            if 'directness' in communication_style:
                directness = communication_style['directness']
                if directness > 0.7:
                    descriptions.append("直接坦率")
                elif directness < 0.3:
                    descriptions.append("委婉含蓄")
            
            if 'humor_usage' in communication_style:
                humor = communication_style['humor_usage']
                if humor > 0.6:
                    descriptions.append("幽默风趣")
            
            if 'emoji_usage' in communication_style:
                emoji = communication_style['emoji_usage']
                if emoji > 0.6:
                    descriptions.append("表情丰富")
            
            return "，".join(descriptions) if descriptions else "普通交流风格"
            
        except Exception as e:
            logger.debug(f"格式化沟通风格失败: {e}")
            return ""
    
    def _format_emotional_tendency(self, emotional_tendency: dict) -> str:
        """
        将情感倾向字典转换为可读描述
        
        Args:
            emotional_tendency: 情感倾向字典
            
        Returns:
            str: 可读的描述文本
        """
        try:
            if not emotional_tendency or not isinstance(emotional_tendency, dict):
                return ""
            
            descriptions = []
            
            # 解析情感倾向特征
            if 'positivity' in emotional_tendency:
                positivity = emotional_tendency['positivity']
                if positivity > 0.7:
                    descriptions.append("积极乐观")
                elif positivity < 0.3:
                    descriptions.append("情绪较低")
            
            if 'stability' in emotional_tendency:
                stability = emotional_tendency['stability']
                if stability > 0.7:
                    descriptions.append("情绪稳定")
                elif stability < 0.3:
                    descriptions.append("情绪波动")
            
            if 'empathy' in emotional_tendency:
                empathy = emotional_tendency['empathy']
                if empathy > 0.6:
                    descriptions.append("善解人意")
            
            if 'expressiveness' in emotional_tendency:
                expressiveness = emotional_tendency['expressiveness']
                if expressiveness > 0.6:
                    descriptions.append("表达丰富")
                elif expressiveness < 0.3:
                    descriptions.append("表达内敛")
            
            if 'dominant_emotion' in emotional_tendency:
                dominant = emotional_tendency['dominant_emotion']
                emotion_map = {
                    'happy': '快乐',
                    'calm': '平静',
                    'excited': '兴奋',
                    'serious': '严肃',
                    'playful': '活泼',
                    'thoughtful': '深思',
                    'caring': '关怀'
                }
                if dominant in emotion_map:
                    descriptions.append(f"偏向{emotion_map[dominant]}")
            
            return "，".join(descriptions) if descriptions else "情感表达平和"
            
        except Exception as e:
            logger.debug(f"格式化情感倾向失败: {e}")
            return ""
