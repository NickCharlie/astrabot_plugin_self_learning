"""
AstrBot 自学习插件 - 智能对话风格学习与人格优化
"""
import os
import json # 导入 json 模块
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter
import astrbot.api.star as star
from astrbot.api.star import register, Context
from astrbot.api import logger, AstrBotConfig
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .config import PluginConfig
from .core.factory import FactoryManager
from .exceptions import SelfLearningError
from .webui import Server, set_plugin_services # 导入 FastAPI 服务器相关

server_instance: Optional[Server] = None # 全局服务器实例

@dataclass
class LearningStats:
    """学习统计信息"""
    total_messages_collected: int = 0
    filtered_messages: int = 0
    style_updates: int = 0
    persona_updates: int = 0
    last_learning_time: Optional[str] = None
    last_persona_update: Optional[str] = None


@register("astrbot_plugin_self_learning", "NickMo", "智能自学习对话插件", "1.0.0", "https://github.com/NickCharlie/astrabot_plugin_self_learning")
class SelfLearningPlugin(star.Star):
    """AstrBot 自学习插件 - 智能学习用户对话风格并优化人格设置"""

    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        
        # 初始化插件配置
        # 获取插件数据目录，并传递给 PluginConfig
        plugin_data_dir = os.path.join(get_astrbot_data_path(), "plugins", "astrabot_plugin_self_learning")
        self.plugin_config = PluginConfig.create_from_config(self.config, data_dir=plugin_data_dir)
        
        # 确保数据目录存在
        os.makedirs(self.plugin_config.data_dir, exist_ok=True)
        
        # 初始化 messages_db_path 和 learning_log_path
        if not self.plugin_config.messages_db_path:
            self.plugin_config.messages_db_path = os.path.join(self.plugin_config.data_dir, "messages.db")
        if not self.plugin_config.learning_log_path:
            self.plugin_config.learning_log_path = os.path.join(self.plugin_config.data_dir, "learning.log")
        
        # 学习统计
        self.learning_stats = LearningStats()
        
        # 初始化服务层
        self._initialize_services()

        # 初始化 Web 服务器
        global server_instance
        if self.plugin_config.enable_web_interface:
            server_instance = Server(port=self.plugin_config.web_interface_port)
            if server_instance:
                logger.info(f"Web 界面已启用，将在 http://{server_instance.host}:{server_instance.port} 启动")
                # 直接启动服务器而不是等待 on_load
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._start_web_server())
                    logger.info("Web 服务器启动任务已创建")
                except Exception as e:
                    logger.error(f"创建 Web 服务器启动任务失败: {e}", exc_info=True)
            else:
                logger.error("Web 界面初始化失败")
        else:
            logger.info("Web 界面未启用")
        
        logger.info("自学习插件初始化完成")

    async def _start_web_server(self):
        """启动Web服务器的异步方法"""
        global server_instance
        if server_instance:
            logger.info("开始启动 Web 服务器...")
            try:
                await server_instance.start()
                logger.info("Web 服务器启动成功")
                
                # 启动数据库管理器
                await self.db_manager.start()
                logger.info("数据库管理器启动完成")
            except Exception as e:
                logger.error(f"Web 服务器启动失败: {e}", exc_info=True)

    def _initialize_services(self):
        """初始化所有服务层组件 - 使用工厂模式"""
        try:
            # 初始化工厂管理器
            self.factory_manager = FactoryManager()
            self.factory_manager.initialize_factories(self.plugin_config, self.context)
            
            # 获取服务工厂
            self.service_factory = self.factory_manager.get_service_factory()
            
            # 使用工厂创建服务
            self.db_manager = self.service_factory.create_database_manager()
            self.message_collector = self.service_factory.create_message_collector()
            self.multidimensional_analyzer = self.service_factory.create_multidimensional_analyzer()
            self.style_analyzer = self.service_factory.create_style_analyzer()
            self.quality_monitor = self.service_factory.create_quality_monitor()
            self.progressive_learning = self.service_factory.create_progressive_learning()
            self.intelligent_responder = self.service_factory.create_intelligent_responder()
            self.ml_analyzer = self.service_factory.create_ml_analyzer()
            self.persona_manager = self.service_factory.create_persona_manager() # 更名为 persona_manager
            
            # 初始化内部组件
            self._setup_internal_components()

            # 将服务实例传递给 Web 服务器模块
            if self.plugin_config.enable_web_interface and server_instance:
                set_plugin_services(
                    self.plugin_config,
                    self.factory_manager, # 传递 factory_manager
                    self.service_factory.create_llm_client() # 传递 LLMClient 实例
                )
            
            logger.info("自学习插件工厂模式服务层初始化完成")
            
        except SelfLearningError as sle:
            logger.error(f"自学习服务初始化失败: {sle}")
            raise # Re-raise as this is an expected initialization failure
        except (TypeError, ValueError) as e: # Catch common initialization errors
            logger.error(f"服务层初始化过程中发生配置或类型错误: {e}", exc_info=True)
            raise SelfLearningError(f"插件初始化失败: {str(e)}") from e
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"服务层初始化过程中发生未知错误: {e}", exc_info=True)
            raise SelfLearningError(f"插件初始化失败: {str(e)}") from e
    
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
        
        # 异步任务管理
        self.background_tasks = set()
        
        # 启动异步任务并追踪
        # 延迟启动学习服务，并传递 group_id
        # 注意：这里需要一个 group_id 来启动学习，对于插件初始化，可以考虑一个默认的全局 group_id
        # 或者在实际消息处理时才启动针对特定 group_id 的学习
        # 暂时不在这里启动全局学习，而是通过命令或消息触发
        # task = asyncio.create_task(self._delayed_start_learning())
        # self.background_tasks.add(task)
        # task.add_done_callback(self.background_tasks.discard) # 任务完成后从集合中移除
    
    async def on_load(self):
        """插件加载时启动 Web 服务器和数据库管理器"""
        logger.info("开始执行 on_load 方法")
        global server_instance
        if self.plugin_config.enable_web_interface and server_instance:
            logger.info(f"准备启动 Web 服务器，地址: {server_instance.host}:{server_instance.port}")
            try:
                await server_instance.start()
                logger.info("Web 服务器启动完成")
            except Exception as e:
                logger.error(f"Web 服务器启动失败: {e}", exc_info=True)
        else:
            if not self.plugin_config.enable_web_interface:
                logger.info("Web 界面被禁用，跳过服务器启动")
            if not server_instance:
                logger.error("Server 实例为 None，无法启动 Web 服务器")
        
        # 启动数据库管理器，确保数据库表被创建
        try:
            await self.db_manager.start()
            logger.info("数据库管理器启动完成")
        except Exception as e:
            logger.error(f"数据库管理器启动失败: {e}", exc_info=True)
        
        logger.info("自学习插件加载完成")

    async def _delayed_start_learning(self, group_id: str):
        """延迟启动学习服务"""
        try:
            await asyncio.sleep(3)  # 等待初始化完成
            await self.service_factory.initialize_all_services() # 确保所有服务初始化完成
            # 启动针对特定 group_id 的渐进式学习
            await self.progressive_learning.start_learning(group_id)
            logger.info(f"自动学习调度器已启动 for group {group_id}")
        except Exception as e:
            logger.error(f"启动学习服务失败 for group {group_id}: {e}")

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
                
            # 收集消息
            await self.message_collector.collect_message({
                'sender_id': sender_id,
                'sender_name': event.get_sender_name(),
                'message': message_text,
                'group_id': group_id, # 使用 group_id
                'timestamp': time.time(),
                'platform': event.get_platform_name()
            })
            
            self.learning_stats.total_messages_collected += 1
            
            # 如果启用实时学习，立即进行筛选
            if self.plugin_config.enable_realtime_learning:
                await self._process_message_realtime(group_id, message_text, sender_id) # 传递 group_id
                
        except Exception as e:
            logger.error(f"消息收集过程中发生未知错误: {e}", exc_info=True)

    async def _process_message_realtime(self, group_id: str, message_text: str, sender_id: str):
        """实时处理消息"""
        try:
            # 使用弱模型筛选消息
            # 获取当前会话的人格描述
            current_persona_description = await self.persona_manager.get_current_persona_description()
            
            if await self.multidimensional_analyzer.filter_message_with_llm(message_text, current_persona_description):
                await self.message_collector.add_filtered_message({
                    'message': message_text,
                    'sender_id': sender_id,
                    'group_id': group_id, # 添加 group_id
                    'timestamp': time.time(),
                    'confidence': 0.8  # 实时筛选置信度
                })
                self.learning_stats.filtered_messages += 1
                
        except Exception as e:
            logger.error(f"实时消息处理过程中发生未知错误: {e}", exc_info=True)

    @filter.command("learning_status")
    async def learning_status_command(self, event: AstrMessageEvent):
        """查看学习状态"""
        try:
            group_id = event.get_group_id() or event.get_sender_id() # 获取当前会话ID
            
            # 获取收集统计
            collector_stats = await self.message_collector.get_statistics(group_id) # 传入 group_id
            
            # 获取当前人格设置
            current_persona_info = await self.persona_manager.get_current_persona(group_id)
            current_persona_name = current_persona_info.get('name', '未知') if current_persona_info else '未知'
            
            # 获取渐进式学习服务的状态
            learning_status = await self.progressive_learning.get_learning_status()
            
            status_info = f"""📚 自学习插件状态报告 (会话ID: {group_id}):

🔧 基础配置:
- 消息抓取: {'✅ 启用' if self.plugin_config.enable_message_capture else '❌ 禁用'}
- 自主学习: {'✅ 启用' if self.plugin_config.enable_auto_learning else '❌ 禁用'}
- 实时学习: {'✅ 启用' if self.plugin_config.enable_realtime_learning else '❌ 禁用'}
- Web界面: {'✅ 启用' if self.plugin_config.enable_web_interface else '❌ 禁用'}

👥 抓取设置:
- 目标QQ: {self.plugin_config.target_qq_list if self.plugin_config.target_qq_list else '全部用户'}
- 当前人格: {current_persona_name}

🤖 模型配置:
- 筛选模型: {self.plugin_config.filter_model_name}
- 提炼模型: {self.plugin_config.refine_model_name}

📊 学习统计 (当前会话):
- 总收集消息: {collector_stats.get('total_messages', 0)}
- 筛选消息: {collector_stats.get('filtered_messages', 0)}  
- 风格更新次数: {learning_status.get('current_session', {}).get('style_updates', 0)}
- 最后学习时间: {learning_status.get('current_session', {}).get('end_time', '从未执行')}

💾 存储统计 (当前会话):
- 原始消息: {collector_stats.get('raw_messages', 0)} 条
- 待处理消息: {collector_stats.get('unprocessed_messages', 0)} 条
- 筛选过的消息: {collector_stats.get('filtered_messages', 0)} 条

⏰ 调度状态 (当前会话): {'🟢 运行中' if learning_status.get('learning_active') else '🔴 已停止'}"""

            yield event.plain_result(status_info.strip())
            
        except Exception as e:
            logger.error(f"获取学习状态失败: {e}", exc_info=True)
            yield event.plain_result(f"状态查询失败: {str(e)}")

    @filter.command("start_learning")
    async def start_learning_command(self, event: AstrMessageEvent):
        """手动启动学习"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            
            if await self.progressive_learning.start_learning(group_id):
                yield event.plain_result(f"✅ 自动学习已启动 for group {group_id}")
            else:
                yield event.plain_result(f"📚 自动学习已在运行中 for group {group_id}")
            
        except Exception as e:
            logger.error(f"启动学习失败: {e}", exc_info=True)
            yield event.plain_result(f"启动失败: {str(e)}")

    @filter.command("stop_learning")
    async def stop_learning_command(self, event: AstrMessageEvent):
        """停止学习"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            
            # ProgressiveLearningService 的 stop_learning 目前没有 group_id 参数
            # 如果需要停止特定 group_id 的学习，ProgressiveLearningService 需要修改
            # 暂时调用全局停止，或者假设 stop_learning 会停止当前活跃的会话
            await self.progressive_learning.stop_learning()
            yield event.plain_result(f"⏹️ 自动学习已停止 for group {group_id}")
            
        except Exception as e:
            logger.error(f"停止学习失败: {e}", exc_info=True)
            yield event.plain_result(f"停止失败: {str(e)}")

    @filter.command("force_learning")  
    async def force_learning_command(self, event: AstrMessageEvent):
        """强制执行一次学习周期"""
        try:
            group_id = event.get_group_id() or event.get_sender_id()
            yield event.plain_result(f"🔄 开始强制学习周期 for group {group_id}...")
            
            # 直接调用 ProgressiveLearningService 的批处理方法
            await self.progressive_learning._execute_learning_batch(group_id)
            
            yield event.plain_result(f"✅ 强制学习周期完成 for group {group_id}")
            
        except Exception as e:
            logger.error(f"强制学习失败: {e}", exc_info=True)
            yield event.plain_result(f"强制学习失败: {str(e)}")

    @filter.command("clear_data")
    async def clear_data_command(self, event: AstrMessageEvent):
        """清空学习数据"""
        try:
            await self.message_collector.clear_all_data()
            
            # 重置统计
            self.learning_stats = LearningStats()
            
            yield event.plain_result("🗑️ 所有学习数据已清空")
            
        except Exception as e: # Consider more specific exceptions if possible
            logger.error(f"清空数据失败: {e}", exc_info=True)
            yield event.plain_result(f"清空数据失败: {str(e)}")

    @filter.command("export_data")
    async def export_data_command(self, event: AstrMessageEvent):
        """导出学习数据"""
        try:
            export_data = await self.message_collector.export_learning_data()
            
            # 生成导出文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_data_export_{timestamp}.json"
            filepath = os.path.join(self.plugin_config.data_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
                
            yield event.plain_result(f"📤 学习数据已导出到: {filepath}")
            
        except Exception as e: # Consider more specific exceptions if possible
            logger.error(f"导出数据失败: {e}", exc_info=True)
            yield event.plain_result(f"导出数据失败: {str(e)}")

    async def terminate(self):
        """插件卸载时的清理工作"""
        try:
            # 停止学习调度器
            if hasattr(self, 'learning_scheduler'):
                await self.learning_scheduler.stop()
                
            # 取消所有后台任务
            for task in list(self.background_tasks): # 使用 list() 避免在迭代时修改集合
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass # 任务已被取消，这是预期行为
                except Exception as e:
                    logger.error(f"取消后台任务时发生错误: {e}", exc_info=True)
            
            # 保存最终状态
            if hasattr(self, 'message_collector'):
                await self.message_collector.save_state()
                
            # 停止 Web 服务器
            global server_instance
            if server_instance:
                await server_instance.stop()
                
            # 保存配置到文件
            with open(os.path.join(self.plugin_config.data_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(self.plugin_config.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info("插件配置已保存")
            
            logger.info("自学习插件已安全卸载")
            
        except Exception as e: # Consider more specific exceptions if possible
            logger.error(f"插件卸载清理失败: {e}", exc_info=True)
