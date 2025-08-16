"""
AstrBot 自学习插件 - 智能对话风格学习与人格优化
基于 voice_to_text 插件架构设计
"""
import os
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter
import astrbot.api.star as star
from astrbot.api.star import register, Context
from astrbot.api import logger, AstrBotConfig

from .config import PluginConfig
from .core.factory import FactoryManager
from .exceptions import SelfLearningError


@dataclass
class LearningStats:
    """学习统计信息"""
    total_messages_collected: int = 0
    filtered_messages: int = 0
    style_updates: int = 0
    persona_updates: int = 0
    last_learning_time: Optional[str] = None
    last_persona_update: Optional[str] = None


@register("self_learning", "NickMo", "智能自学习对话插件", "1.0.0", "https://github.com/NickCharlie/astrabot_plugin_self_learning")
class SelfLearningPlugin(star.Star):
    """AstrBot 自学习插件 - 智能学习用户对话风格并优化人格设置"""

    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        
        # 初始化插件配置
        self.plugin_config = PluginConfig.create_from_config(self.config)
        
        # 学习统计
        self.learning_stats = LearningStats()
        
        # 初始化服务层
        self._initialize_services()
        
        logger.info("自学习插件初始化完成")

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
            self.backup_manager = self.service_factory.create_persona_manager()
            
            # 初始化所有服务
            asyncio.create_task(self.service_factory.initialize_all_services())
            
            # 初始化内部组件
            self._setup_internal_components()
            
            logger.info("自学习插件工厂模式服务层初始化完成")
            
        except Exception as e:
            logger.error(f"服务层初始化失败: {e}")
            raise SelfLearningError(f"插件初始化失败: {str(e)}") from e
    
    def _setup_internal_components(self):
        """设置内部组件"""
        # QQ号过滤器
        self.qq_filter = self._create_qq_filter()
        
        # 消息过滤器
        self.message_filter = self._create_message_filter()
        
        # 人格更新器
        self.persona_updater = self._create_persona_updater()
        
        # 学习调度器
        self.learning_scheduler = self._create_learning_scheduler()
        
        # 异步任务管理
        self.background_tasks = set()
        
        # 启动异步任务
        self._schedule_background_tasks()
    
    def _create_qq_filter(self):
        """创建QQ号过滤器"""
        class QQFilter:
            def __init__(self, target_qq_list):
                self.target_qq_list = target_qq_list
            
            def should_collect_message(self, sender_id: str) -> bool:
                if not self.target_qq_list:  # 空列表表示收集所有
                    return True
                return sender_id in self.target_qq_list
        
        return QQFilter(self.plugin_config.target_qq_list)
    
    def _create_message_filter(self):
        """创建消息过滤器"""
        class MessageFilter:
            def __init__(self, config, context):
                self.config = config
                self.context = context
            
            async def is_suitable_for_learning(self, message: str) -> bool:
                # 基础长度检查
                if len(message) < self.config.message_min_length:
                    return False
                if len(message) > self.config.message_max_length:
                    return False
                
                # 简单内容过滤
                if message.strip() in ['', '???', '。。。', '...']:
                    return False
                
                return True
        
        return MessageFilter(self.plugin_config, self.context)
    
    def _create_persona_updater(self):
        """创建人格更新器"""
        class PersonaUpdater:
            def __init__(self, context, backup_manager):
                self.context = context
                self.backup_manager = backup_manager
            
            async def update_persona_with_style(self, style_analysis, filtered_messages) -> bool:
                try:
                    # 这里应该实现实际的人格更新逻辑
                    logger.info("人格更新功能待完善实现")
                    return True
                except Exception as e:
                    logger.error(f"人格更新失败: {e}")
                    return False
        
        return PersonaUpdater(self.context, self.backup_manager)
    
    def _create_learning_scheduler(self):
        """创建学习调度器"""
        class LearningScheduler:
            def __init__(self, plugin_instance):
                self.plugin = plugin_instance
                self.is_running = False
                self._task = None
            
            def start(self):
                if not self.is_running:
                    self.is_running = True
                    self._task = asyncio.create_task(self._learning_loop())
            
            async def stop(self):
                if self.is_running:
                    self.is_running = False
                    if self._task:
                        self._task.cancel()
                        try:
                            await self._task
                        except asyncio.CancelledError:
                            pass
            
            async def _learning_loop(self):
                while self.is_running:
                    try:
                        await asyncio.sleep(self.plugin.plugin_config.learning_interval_hours * 3600)
                        if self.is_running:
                            await self.plugin._perform_learning_cycle()
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"学习循环异常: {e}")
        
        return LearningScheduler(self)
    
    def _schedule_background_tasks(self):
        """安排后台任务"""
        if self.plugin_config.enable_auto_learning:
            task = asyncio.create_task(self._delayed_start_learning())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
    
    async def _delayed_start_learning(self):
        """延迟启动学习服务"""
        try:
            await asyncio.sleep(3)  # 等待初始化完成
            self.learning_scheduler.start()
            logger.info("自动学习调度器已启动")
        except Exception as e:
            logger.error(f"启动学习服务失败: {e}")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent, context=None):
        """监听所有消息，收集用户对话数据"""
        
        # 检查是否启用消息抓取
        if not self.plugin_config.enable_message_capture:
            return
            
        try:
            # QQ号过滤
            sender_id = event.get_sender_id()
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
                'group_id': event.get_group_id(),
                'timestamp': time.time(),
                'platform': event.get_platform_name()
            })
            
            self.learning_stats.total_messages_collected += 1
            
            # 如果启用实时学习，立即进行筛选
            if self.plugin_config.enable_realtime_learning:
                await self._process_message_realtime(message_text, sender_id)
                
        except Exception as e:
            logger.error(f"消息收集失败: {e}")

    async def _process_message_realtime(self, message_text: str, sender_id: str):
        """实时处理消息"""
        try:
            # 使用弱模型筛选消息
            if await self.message_filter.is_suitable_for_learning(message_text):
                await self.message_collector.add_filtered_message({
                    'message': message_text,
                    'sender_id': sender_id,
                    'timestamp': time.time(),
                    'confidence': 0.8  # 实时筛选置信度
                })
                self.learning_stats.filtered_messages += 1
                
        except Exception as e:
            logger.error(f"实时消息处理失败: {e}")

    async def _perform_learning_cycle(self):
        """执行完整的学习周期"""
        try:
            logger.info("开始执行自学习周期...")
            
            # 1. 获取待处理的消息
            raw_messages = await self.message_collector.get_unprocessed_messages()
            if not raw_messages:
                logger.info("没有待处理的消息")
                return
                
            logger.info(f"开始处理 {len(raw_messages)} 条消息")
            
            # 2. 使用弱模型筛选消息
            filtered_messages = []
            for msg in raw_messages:
                if await self.message_filter.is_suitable_for_learning(msg['message']):
                    filtered_messages.append(msg)
                    
            logger.info(f"筛选出 {len(filtered_messages)} 条适合学习的消息")
            self.learning_stats.filtered_messages += len(filtered_messages)
            
            if not filtered_messages:
                return
                
            # 3. 使用强模型分析对话风格
            style_analysis = await self.style_analyzer.analyze_conversation_style(
                filtered_messages
            )
            
            if not style_analysis:
                logger.warning("风格分析失败")
                return
                
            # 4. 更新人格和对话风格
            update_success = await self.persona_updater.update_persona_with_style(
                style_analysis,
                filtered_messages
            )
            
            if update_success:
                self.learning_stats.style_updates += 1
                self.learning_stats.persona_updates += 1
                self.learning_stats.last_learning_time = datetime.now().isoformat()
                self.learning_stats.last_persona_update = datetime.now().isoformat()
                
            # 5. 标记消息为已处理
            await self.message_collector.mark_messages_processed(
                [msg['id'] for msg in raw_messages if 'id' in msg]
            )
            
            logger.info("自学习周期完成")
            
        except Exception as e:
            logger.error(f"学习周期执行失败: {e}")

    @filter.command("learning_status")
    async def learning_status_command(self, event: AstrMessageEvent):
        """查看学习状态"""
        try:
            # 获取收集统计
            collector_stats = await self.message_collector.get_statistics()
            
            # 获取当前人格设置
            current_persona = self.context.get_using_provider().curr_personality.name if self.context.get_using_provider() else "未知"
            
            status_info = f"""📚 自学习插件状态报告:

🔧 基础配置:
- 消息抓取: {'✅ 启用' if self.plugin_config.enable_message_capture else '❌ 禁用'}
- 自主学习: {'✅ 启用' if self.plugin_config.enable_auto_learning else '❌ 禁用'}
- 实时学习: {'✅ 启用' if self.plugin_config.enable_realtime_learning else '❌ 禁用'}
- Web界面: {'✅ 启用' if self.plugin_config.enable_web_interface else '❌ 禁用'}

👥 抓取设置:
- 目标QQ: {self.plugin_config.target_qq_list if self.plugin_config.target_qq_list else '全部用户'}
- 当前人格: {current_persona}

🤖 模型配置:
- 筛选模型: {self.plugin_config.filter_model_name}
- 提炼模型: {self.plugin_config.refine_model_name}

📊 学习统计:
- 总收集消息: {self.learning_stats.total_messages_collected}
- 筛选消息: {self.learning_stats.filtered_messages}  
- 风格更新次数: {self.learning_stats.style_updates}
- 人格更新次数: {self.learning_stats.persona_updates}
- 最后学习时间: {self.learning_stats.last_learning_time or '从未执行'}

💾 存储统计:
- 原始消息: {collector_stats.get('raw_messages', 0)} 条
- 待处理消息: {collector_stats.get('unprocessed_messages', 0)} 条
- 筛选过的消息: {collector_stats.get('filtered_messages', 0)} 条

⏰ 调度状态: {'🟢 运行中' if self.learning_scheduler.is_running else '🔴 已停止'}"""

            yield event.plain_result(status_info.strip())
            
        except Exception as e:
            logger.error(f"获取学习状态失败: {e}")
            yield event.plain_result(f"状态查询失败: {str(e)}")

    @filter.command("start_learning")
    async def start_learning_command(self, event: AstrMessageEvent):
        """手动启动学习"""
        try:
            if self.learning_scheduler.is_running:
                yield event.plain_result("📚 自动学习已在运行中")
                return
                
            self.learning_scheduler.start()
            yield event.plain_result("✅ 自动学习已启动")
            
        except Exception as e:
            logger.error(f"启动学习失败: {e}")
            yield event.plain_result(f"启动失败: {str(e)}")

    @filter.command("stop_learning")
    async def stop_learning_command(self, event: AstrMessageEvent):
        """停止学习"""
        try:
            if not self.learning_scheduler.is_running:
                yield event.plain_result("📚 自动学习未运行")
                return
                
            await self.learning_scheduler.stop()
            yield event.plain_result("⏹️ 自动学习已停止")
            
        except Exception as e:
            logger.error(f"停止学习失败: {e}")
            yield event.plain_result(f"停止失败: {str(e)}")

    @filter.command("force_learning")  
    async def force_learning_command(self, event: AstrMessageEvent):
        """强制执行一次学习周期"""
        try:
            yield event.plain_result("🔄 开始强制学习周期...")
            await self._perform_learning_cycle()
            yield event.plain_result("✅ 强制学习周期完成")
            
        except Exception as e:
            logger.error(f"强制学习失败: {e}")
            yield event.plain_result(f"强制学习失败: {str(e)}")

    @filter.command("clear_data")
    async def clear_data_command(self, event: AstrMessageEvent):
        """清空学习数据"""
        try:
            await self.message_collector.clear_all_data()
            
            # 重置统计
            self.learning_stats = LearningStats()
            
            yield event.plain_result("🗑️ 所有学习数据已清空")
            
        except Exception as e:
            logger.error(f"清空数据失败: {e}")
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
            
        except Exception as e:
            logger.error(f"导出数据失败: {e}")
            yield event.plain_result(f"导出数据失败: {str(e)}")

    async def terminate(self):
        """插件卸载时的清理工作"""
        try:
            # 停止学习调度器
            if hasattr(self, 'learning_scheduler'):
                await self.learning_scheduler.stop()
                
            # 保存最终状态
            if hasattr(self, 'message_collector'):
                await self.message_collector.save_state()
                
            logger.info("自学习插件已安全卸载")
            
        except Exception as e:
            logger.error(f"插件卸载清理失败: {e}")
