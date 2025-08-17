"""
服务工厂 - 工厂模式实现，避免循环导入
"""
from typing import Dict, Any, Optional
import logging
import asyncio

from astrbot.api.star import Context

from .interfaces import (
    IServiceFactory, IMessageCollector, IStyleAnalyzer, ILearningStrategy,
    IQualityMonitor, IPersonaManager, IMLAnalyzer, IIntelligentResponder,
    LearningStrategyType, ServiceError
)
from .patterns import StrategyFactory, ServiceRegistry, EventBus
from ..config import PluginConfig
from .llm_client import LLMClient # 导入 LLMClient


class ServiceFactory(IServiceFactory):
    """主要服务工厂 - 创建和管理所有服务实例"""
    
    def __init__(self, config: PluginConfig, context: Context):
        self.config = config
        self.context = context
        self._logger = logging.getLogger(self.__class__.__name__)
        self._registry = ServiceRegistry()
        self._event_bus = EventBus()
        
        # 服务实例缓存
        self._service_cache: Dict[str, Any] = {}
        
        # 初始化 LLM 客户端
        self._llm_client = LLMClient(context, context.get_astrbot_config())
    
    def create_message_collector(self) -> IMessageCollector:
        """创建消息收集器"""
        cache_key = "message_collector"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            # 动态导入避免循环依赖
            from ..services.message_collector import MessageCollectorService
            
            service = MessageCollectorService(self.config, self.context, self._llm_client, self.create_database_manager()) # 传递 LLMClient 和 DatabaseManager
            self._service_cache[cache_key] = service
            self._registry.register_service("message_collector", service)
            
            self._logger.info("创建消息收集器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入消息收集器失败: {e}")
            raise ServiceError(f"创建消息收集器失败: {str(e)}")
    
    def create_style_analyzer(self) -> IStyleAnalyzer:
        """创建风格分析器"""
        cache_key = "style_analyzer"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.style_analyzer import StyleAnalyzerService
            
            service = StyleAnalyzerService(self.config, self.context, self._llm_client) # 传递 LLMClient
            self._service_cache[cache_key] = service
            self._registry.register_service("style_analyzer", service)
            
            self._logger.info("创建风格分析器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入风格分析器失败: {e}")
            raise ServiceError(f"创建风格分析器失败: {str(e)}")
    
    def create_learning_strategy(self, strategy_type: str) -> ILearningStrategy:
        """创建学习策略"""
        try:
            # 转换字符串为枚举
            if isinstance(strategy_type, str):
                strategy_enum = LearningStrategyType(strategy_type)
            else:
                strategy_enum = strategy_type
            
            # 使用策略工厂创建
            strategy_config = {
                'batch_size': self.config.max_messages_per_batch,
                'min_messages': self.config.min_messages_for_learning,
                'min_interval_hours': self.config.learning_interval_hours
            }
            
            strategy = StrategyFactory.create_strategy(strategy_enum, strategy_config)
            self._logger.info(f"创建学习策略成功: {strategy_type}")
            
            return strategy
            
        except ValueError as e:
            self._logger.error(f"不支持的策略类型: {strategy_type}")
            raise ServiceError(f"创建学习策略失败: {str(e)}")
    
    def create_quality_monitor(self) -> IQualityMonitor:
        """创建质量监控器"""
        cache_key = "quality_monitor"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.learning_quality_monitor import LearningQualityMonitor
            
            service = LearningQualityMonitor(self.config, self.context)
            self._service_cache[cache_key] = service
            self._registry.register_service("quality_monitor", service)
            
            self._logger.info("创建质量监控器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入质量监控器失败: {e}")
            raise ServiceError(f"创建质量监控器失败: {str(e)}")
    
    def create_database_manager(self):
        """创建数据库管理器"""
        cache_key = "database_manager"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.database_manager import DatabaseManager as DBManager

            
            service = DBManager(self.config, self.context)
            self._service_cache[cache_key] = service
            self._registry.register_service("database_manager", service)
            
            self._logger.info("创建数据库管理器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入数据库管理器失败: {e}")
            raise ServiceError(f"创建数据库管理器失败: {str(e)}")
    
    def create_ml_analyzer(self) -> IMLAnalyzer:
        """创建ML分析器"""
        cache_key = "ml_analyzer"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.ml_analyzer import LightweightMLAnalyzer
            
            # 需要数据库管理器
            db_manager = self.create_database_manager()
            
            service = LightweightMLAnalyzer(self.config, db_manager)
            self._service_cache[cache_key] = service
            
            self._logger.info("创建ML分析器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入ML分析器失败: {e}")
            raise ServiceError(f"创建ML分析器失败: {str(e)}")
    
    def create_intelligent_responder(self) -> IIntelligentResponder:
        """创建智能回复器"""
        cache_key = "intelligent_responder"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.intelligent_responder import IntelligentResponder
            
            # 需要数据库管理器
            db_manager = self.create_database_manager()
            
            service = IntelligentResponder(self.config, self.context, db_manager, self._llm_client) # 传递 LLMClient
            self._service_cache[cache_key] = service
            
            self._logger.info("创建智能回复器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入智能回复器失败: {e}")
            raise ServiceError(f"创建智能回复器失败: {str(e)}")
    
    def create_persona_manager(self) -> IPersonaManager:
        """创建人格管理器"""
        cache_key = "persona_manager"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            # 创建备份管理器
            backup_manager = self.create_persona_backup_manager()
            
            # 创建人格更新器 (现在由 ServiceFactory 内部创建)
            service = self.create_persona_updater()
            self._service_cache[cache_key] = service
            
            self._logger.info("创建人格管理器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入人格管理器失败: {e}")
            raise ServiceError(f"创建人格管理器失败: {str(e)}")
    
    def create_multidimensional_analyzer(self):
        """创建多维度分析器"""
        cache_key = "multidimensional_analyzer"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.multidimensional_analyzer import MultidimensionalAnalyzer
            
            service = MultidimensionalAnalyzer(self.config, self.context, self._llm_client) # 传递 LLMClient
            self._service_cache[cache_key] = service
            
            self._logger.info("创建多维度分析器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入多维度分析器失败: {e}")
            raise ServiceError(f"创建多维度分析器失败: {str(e)}")
    
    def create_progressive_learning(self):
        """创建渐进式学习服务"""
        cache_key = "progressive_learning"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.progressive_learning import ProgressiveLearningService
            
            service = ProgressiveLearningService(self.config, self.context, self._llm_client) # 传递 LLMClient
            self._service_cache[cache_key] = service
            self._registry.register_service("progressive_learning", service)
            
            self._logger.info("创建渐进式学习服务成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入渐进式学习服务失败: {e}")
            raise ServiceError(f"创建渐进式学习服务失败: {str(e)}")
    
    def create_persona_backup_manager(self):
        """创建人格备份管理器"""
        cache_key = "persona_backup_manager"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.persona_backup_manager import PersonaBackupManager
            db_manager = self.create_database_manager()
            service = PersonaBackupManager(self.config, self.context, db_manager)
            self._service_cache[cache_key] = service
            self._registry.register_service("persona_backup_manager", service)
            self._logger.info("创建人格备份管理器成功")
            return service
        except ImportError as e:
            self._logger.error(f"导入人格备份管理器失败: {e}")
            raise ServiceError(f"创建人格备份管理器失败: {str(e)}")

    def create_persona_updater(self) -> IPersonaManager:
        """创建人格更新器"""
        cache_key = "persona_updater"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.persona_updater import PersonaUpdater
            backup_manager = self.create_persona_backup_manager()
            service = PersonaUpdater(self.config, self.context, backup_manager, self._llm_client)
            self._service_cache[cache_key] = service
            self._registry.register_service("persona_updater", service)
            self._logger.info("创建人格更新器成功")
            return service
        except ImportError as e:
            self._logger.error(f"导入人格更新器失败: {e}")
            raise ServiceError(f"创建人格更新器失败: {str(e)}")

    def get_service_registry(self) -> ServiceRegistry:
        """获取服务注册表"""
        return self._registry
    
    def get_event_bus(self) -> EventBus:
        """获取事件总线"""
        return self._event_bus
    
    async def initialize_all_services(self) -> bool:
        """初始化所有服务"""
        self._logger.info("开始初始化所有服务")
        
        try:
            # 按依赖顺序创建服务
            self.create_database_manager()
            self.create_message_collector()
            self.create_style_analyzer()
            self.create_quality_monitor()
            self.create_ml_analyzer()
            self.create_intelligent_responder()
            self.create_persona_manager()
            self.create_multidimensional_analyzer()
            self.create_progressive_learning()
            
            # 启动所有注册的服务
            success = await self._registry.start_all_services()
            
            if success:
                self._logger.info("所有服务初始化成功")
            else:
                self._logger.error("部分服务初始化失败")
            
            return success
            
        except Exception as e:
            self._logger.error(f"服务初始化异常: {e}")
            return False
    
    async def shutdown_all_services(self) -> bool:
        """关闭所有服务"""
        self._logger.info("开始关闭所有服务")
        
        try:
            success = await self._registry.stop_all_services()
            
            # 清理缓存
            self._service_cache.clear()
            
            if success:
                self._logger.info("所有服务关闭成功")
            else:
                self._logger.error("部分服务关闭失败")
            
            return success
            
        except Exception as e:
            self._logger.error(f"服务关闭异常: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, str]:
        """获取所有服务状态"""
        return self._registry.get_service_status()
    
    def clear_cache(self):
        """清理服务缓存"""
        self._service_cache.clear()
        self._logger.info("服务缓存已清理")


class ComponentFactory:
    """组件工厂 - 创建轻量级组件"""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def create_qq_filter(self):
        """创建QQ号过滤器"""
        class QQFilter:
            def __init__(self, target_qq_list):
                self.target_qq_list = target_qq_list or []
            
            def should_collect_message(self, sender_id: str) -> bool:
                if not self.target_qq_list:  # 空列表表示收集所有
                    return True
                return sender_id in self.target_qq_list
        
        return QQFilter(self.config.target_qq_list)
    
    def create_message_filter(self, context: Context, llm_client: LLMClient): # 接收 llm_client
        """创建消息过滤器"""
        class MessageFilter:
            def __init__(self, config: PluginConfig, context: Context, llm_client: LLMClient):
                self.config = config
                self.context = context
                self.llm_client = llm_client
                self._logger = logging.getLogger(self.__class__.__name__)
            
            async def is_suitable_for_learning(self, message: str) -> bool:
                # 基础长度检查
                if len(message) < self.config.message_min_length:
                    return False
                if len(message) > self.config.message_max_length:
                    return False
                
                # 简单内容过滤
                if message.strip() in ['', '???', '。。。', '...']:
                    return False
                
                # 使用 LLM 进行初步筛选
                try:
                    current_persona = self.context.get_using_provider().curr_personality.prompt if self.context.get_using_provider() else "默认人格"
                    
                    prompt = f"""请判断以下消息是否与当前人格匹配，特征鲜明，且具有学习意义。
当前人格描述: {current_persona}
消息内容: "{message}"

请以 JSON 格式返回判断结果，包含 'suitable' (布尔值) 和 'confidence' (0.0-1.0 之间的浮点数)。
例如: {{"suitable": true, "confidence": 0.9}}"""
                    
                    response = await self.llm_client.chat_completion(
                        provider_id=self.config.filter_provider_id,
                        model_name=self.config.filter_model_name,
                        prompt=prompt,
                        system_prompt="你是一个消息筛选助手，请根据消息内容和当前人格描述，判断消息是否适合用于学习和优化人格。",
                        temperature=0.2 # 降低温度以获得更确定的结果
                    )
                    
                    if response and response.get('text'):
                        try:
                            llm_result = json.loads(response['text'])
                            suitable = llm_result.get('suitable', False)
                            confidence = llm_result.get('confidence', 0.0)
                            
                            self._logger.debug(f"LLM 筛选结果: message='{message}', suitable={suitable}, confidence={confidence}")
                            
                            # 结合置信度阈值进行判断
                            return suitable and confidence >= self.config.confidence_threshold
                        except json.JSONDecodeError:
                            self._logger.warning(f"LLM 返回结果不是有效的 JSON: {response['text']}")
                            return False # LLM 返回无效 JSON，认为不适合
                    else:
                        self._logger.warning("LLM 筛选未返回有效结果。")
                        return False # LLM 未返回结果，认为不适合
                except Exception as e:
                    self._logger.error(f"LLM 筛选消息失败: {e}")
                    return False # LLM 调用失败，认为不适合
        
        return MessageFilter(self.config, context, llm_client) # 传递 llm_client
    
    def create_learning_scheduler(self, plugin_instance):
        """创建学习调度器"""
        class LearningScheduler:
            def __init__(self, plugin_instance):
                self.plugin = plugin_instance
                self.is_running = False
                self._task = None
                self._logger = logging.getLogger(f"{self.__class__.__name__}")
            
            def start(self):
                if not self.is_running:
                    self.is_running = True
                    self._task = asyncio.create_task(self._learning_loop())
                    self._logger.info("学习调度器已启动")
            
            async def stop(self):
                if self.is_running:
                    self.is_running = False
                    if self._task:
                        self._task.cancel()
                        try:
                            await self._task
                        except asyncio.CancelledError:
                            pass
                    self._logger.info("学习调度器已停止")
            
            async def _learning_loop(self):
                while self.is_running:
                    try:
                        interval_seconds = self.plugin.config.learning_interval_hours * 3600
                        await asyncio.sleep(interval_seconds)
                        
                        if self.is_running and hasattr(self.plugin, '_perform_learning_cycle'):
                            await self.plugin._perform_learning_cycle()
                            
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        self._logger.error(f"学习循环异常: {e}")
                        await asyncio.sleep(60)  # 错误后等待1分钟再重试
        
        return LearningScheduler(plugin_instance)
    
    def create_persona_updater(self, context: Context, backup_manager):
        """创建人格更新器"""
        class PersonaUpdater:
            def __init__(self, context, backup_manager):
                self.context = context
                self.backup_manager = backup_manager
                self._logger = logging.getLogger(self.__class__.__name__)
            
            async def update_persona_with_style(self, style_analysis, filtered_messages) -> bool:
                try:
                    # 在更新前创建备份
                    backup_id = await self.backup_manager.create_backup_before_update(
                        "default",
                        f"Style update with {len(filtered_messages)} messages"
                    )
                    self._logger.info(f"创建人格备份: {backup_id}")
                    
                    # 这里应该实现实际的人格更新逻辑
                    # 调用AstrBot框架的人格更新API
                    provider = self.context.get_using_provider()
                    if provider and hasattr(provider, 'curr_personality'):
                        # 更新人格描述
                        if 'enhanced_prompt' in style_analysis:
                            provider.curr_personality.prompt = style_analysis['enhanced_prompt']
                        
                        # 添加对话样本到模仿列表
                        if hasattr(provider, 'mood_imitation_dialogs'):
                            for msg in filtered_messages[-5:]:  # 添加最近5条优质消息
                                if msg.get('message') not in provider.mood_imitation_dialogs:
                                    provider.mood_imitation_dialogs.append(msg.get('message'))
                    
                    self._logger.info("人格更新成功")
                    return True
                    
                except Exception as e:
                    self._logger.error(f"人格更新失败: {e}")
                    return False
        
        return PersonaUpdater(context, backup_manager)


# 全局工厂实例管理器
class FactoryManager:
    """工厂管理器 - 单例模式管理所有工厂"""
    
    _instance = None
    _service_factory: Optional[ServiceFactory] = None
    _component_factory: Optional[ComponentFactory] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize_factories(self, config: PluginConfig, context: Context):
        """初始化工厂"""
        self._service_factory = ServiceFactory(config, context)
        self._component_factory = ComponentFactory(config)
    
    def get_service_factory(self) -> ServiceFactory:
        """获取服务工厂"""
        if self._service_factory is None:
            raise ServiceError("服务工厂未初始化")
        return self._service_factory
    
    def get_component_factory(self) -> ComponentFactory:
        """获取组件工厂"""
        if self._component_factory is None:
            raise ServiceError("组件工厂未初始化")
        return self._component_factory
    
    async def cleanup(self):
        """清理所有工厂"""
        if self._service_factory:
            await self._service_factory.shutdown_all_services()
            self._service_factory.clear_cache()
        
        self._service_factory = None
        self._component_factory = None
