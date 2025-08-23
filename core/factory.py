"""
服务工厂 - 工厂模式实现，避免循环导入
"""
from typing import Dict, Any, Optional
import asyncio
import json # 导入json模块，因为MessageFilter中使用了

from astrbot.api.star import Context
from astrbot.api import logger # 使用框架提供的logger

from .interfaces import (
    IServiceFactory, IMessageCollector, IStyleAnalyzer, ILearningStrategy,
    IQualityMonitor, IPersonaManager, IPersonaUpdater, IMLAnalyzer, IIntelligentResponder,
    LearningStrategyType
)
from .patterns import StrategyFactory, ServiceRegistry, EventBus
from ..config import PluginConfig
from .framework_llm_adapter import FrameworkLLMAdapter # 导入框架LLM适配器
from ..exceptions import ServiceError # 从 exceptions.py 导入 ServiceError
from ..statics import prompts # 导入 prompts 模块
from ..utils.json_utils import safe_parse_llm_json


class ServiceFactory(IServiceFactory):
    """主要服务工厂 - 创建和管理所有服务实例"""
    
    def __init__(self, config: PluginConfig, context: Context):
        self.config = config
        self.context = context
        self._logger = logger
        self._registry = ServiceRegistry()
        self._event_bus = EventBus()
        
        # 服务实例缓存
        self._service_cache: Dict[str, Any] = {}
        
        # 框架适配器
        self._framework_llm_adapter: Optional[FrameworkLLMAdapter] = None
    
    # LLMClient已弃用，使用FrameworkLLMAdapter代替

    def create_framework_llm_adapter(self) -> FrameworkLLMAdapter:
        """创建或获取框架LLM适配器"""
        if self._framework_llm_adapter is None:
            self._framework_llm_adapter = FrameworkLLMAdapter(self.context)
            self._framework_llm_adapter.initialize_providers(self.config)
            self._logger.info("框架LLM适配器初始化成功")
        return self._framework_llm_adapter

    def get_prompts(self) -> Any:
        """获取 Prompt 静态数据"""
        return prompts

    def create_message_collector(self) -> IMessageCollector:
        """创建消息收集器"""
        cache_key = "message_collector"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            # 动态导入避免循环依赖
            from ..services.message_collector import MessageCollectorService
            
            service = MessageCollectorService(self.config, self.context, self.create_database_manager()) # 传递 DatabaseManager
            self._service_cache[cache_key] = service
            self._registry.register_service("message_collector", service)
            
            self._logger.info("创建消息收集器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入消息收集器失败: {e}", exc_info=True)
            raise ServiceError(f"创建消息收集器失败: {str(e)}")
    
    def create_style_analyzer(self) -> IStyleAnalyzer:
        """创建风格分析器"""
        cache_key = "style_analyzer"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.style_analyzer import StyleAnalyzerService
            
            # 传递 DatabaseManager 和框架适配器
            service = StyleAnalyzerService(
                self.config, 
                self.context, 
                self.create_database_manager(),
                llm_adapter=self.create_framework_llm_adapter(),  # 使用框架适配器
                prompts=self.get_prompts()  # 传递 prompts
            ) 
            self._service_cache[cache_key] = service
            self._registry.register_service("style_analyzer", service)
            
            self._logger.info("创建风格分析器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入风格分析器失败: {e}", exc_info=True)
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
            self._logger.error(f"不支持的策略类型: {strategy_type}", exc_info=True)
            raise ServiceError(f"创建学习策略失败: {str(e)}")
    
    def create_quality_monitor(self) -> IQualityMonitor:
        """创建质量监控器"""
        cache_key = "quality_monitor"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.learning_quality_monitor import LearningQualityMonitor
            
            service = LearningQualityMonitor(
                self.config, 
                self.context, 
                llm_adapter=self.create_framework_llm_adapter(),  # 使用框架适配器
                prompts=self.get_prompts()  # 传递 prompts
            ) 
            self._service_cache[cache_key] = service
            self._registry.register_service("quality_monitor", service)
            
            self._logger.info("创建质量监控器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入质量监控器失败: {e}", exc_info=True)
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
            self._logger.error(f"导入数据库管理器失败: {e}", exc_info=True)
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
            
            # 通过工厂方法创建服务，LLMClient已弃用
            refine_llm_client = None # LLMClient已弃用
            reinforce_llm_client = None # LLMClient已弃用

            service = LightweightMLAnalyzer(
                self.config, 
                db_manager, 
                llm_adapter=self.create_framework_llm_adapter(),  # 使用框架适配器
                prompts=self.get_prompts() # 传递 prompts
            )
            self._service_cache[cache_key] = service
            
            self._logger.info("创建ML分析器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入ML分析器失败: {e}", exc_info=True)
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
            
            service = IntelligentResponder(
                self.config, 
                self.context, 
                db_manager, 
                llm_adapter=self.create_framework_llm_adapter(), # 传递框架适配器
                prompts=self.get_prompts() # 传递 prompts
            )
            self._service_cache[cache_key] = service
            
            self._logger.info("创建智能回复器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入智能回复器失败: {e}", exc_info=True)
            raise ServiceError(f"创建智能回复器失败: {str(e)}")
    
    def create_persona_manager(self) -> IPersonaManager:
        """创建人格管理器"""
        cache_key = "persona_manager"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.persona_manager import PersonaManagerService # 导入 PersonaManagerService
            
            # 创建依赖的服务
            persona_updater = self.create_persona_updater()
            persona_backup_manager = self.create_persona_backup_manager()
            
            service = PersonaManagerService(self.config, self.context, persona_updater, persona_backup_manager)
            self._service_cache[cache_key] = service
            self._registry.register_service("persona_manager", service) # 注册服务
            
            self._logger.info("创建人格管理器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入人格管理器失败: {e}", exc_info=True)
            raise ServiceError(f"创建人格管理器失败: {str(e)}")
    
    def create_multidimensional_analyzer(self):
        """创建多维度分析器"""
        cache_key = "multidimensional_analyzer"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.multidimensional_analyzer import MultidimensionalAnalyzer
            
            db_manager = self.create_database_manager() # 获取 DatabaseManager 实例
            
            # 使用框架LLM适配器替代自定义LLMClient
            llm_adapter = self.create_framework_llm_adapter()

            service = MultidimensionalAnalyzer(
                self.config, 
                db_manager, 
                self.context,
                llm_adapter=llm_adapter,  # 传递框架适配器
                prompts=self.get_prompts() # 传递 prompts
            )
            self._service_cache[cache_key] = service
            
            self._logger.info("创建多维度分析器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入多维度分析器失败: {e}", exc_info=True)
            raise ServiceError(f"创建多维度分析器失败: {str(e)}")

    def create_progressive_learning(self):
        """创建渐进式学习服务"""
        cache_key = "progressive_learning"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.progressive_learning import ProgressiveLearningService
            
            # Directly pass the database manager
            db_manager = self.create_database_manager()
            
            service = ProgressiveLearningService(
                self.config, 
                self.context,
                db_manager=db_manager, # 传递 db_manager 实例
                message_collector=self.create_message_collector(),
                multidimensional_analyzer=self.create_multidimensional_analyzer(),
                style_analyzer=self.create_style_analyzer(),
                quality_monitor=self.create_quality_monitor(),
                persona_manager=self.create_persona_manager(), # 传递 persona_manager 实例
                ml_analyzer=self.create_ml_analyzer(), # 传递 ml_analyzer 实例
                prompts=self.get_prompts() # 传递 prompts
            )
            self._service_cache[cache_key] = service
            self._registry.register_service("progressive_learning", service)
            
            self._logger.info("创建渐进式学习服务成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入渐进式学习服务失败: {e}", exc_info=True)
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
            self._logger.error(f"导入人格备份管理器失败: {e}", exc_info=True)
            raise ServiceError(f"创建人格备份管理器失败: {str(e)}")

    def create_temporary_persona_updater(self):
        """创建临时人格更新器"""
        cache_key = "temporary_persona_updater"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.temporary_persona_updater import TemporaryPersonaUpdater
            
            # 获取依赖的服务
            persona_updater = self.create_persona_updater()
            backup_manager = self.create_persona_backup_manager()
            llm_client = None # LLMClient已弃用
            db_manager = self.create_database_manager()
            
            service = TemporaryPersonaUpdater(
                self.config,
                self.context,
                persona_updater,
                backup_manager,
                db_manager,
                llm_client  # 现在是可选参数
            )
            self._service_cache[cache_key] = service
            self._registry.register_service("temporary_persona_updater", service)
            
            self._logger.info("创建临时人格更新器成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入临时人格更新器失败: {e}", exc_info=True)
            raise ServiceError(f"创建临时人格更新器失败: {str(e)}")

    def create_persona_updater(self) -> IPersonaUpdater: # 修改返回类型为 IPersonaUpdater
        """创建人格更新器"""
        cache_key = "persona_updater"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.persona_updater import PersonaUpdater
            backup_manager = self.create_persona_backup_manager()
            service = PersonaUpdater(
                self.config, 
                self.context, 
                backup_manager, 
                None,  # LLMClient已弃用
                self.create_database_manager()  # 传递正确的db_manager
            )
            self._service_cache[cache_key] = service
            self._registry.register_service("persona_updater", service)
            self._logger.info("创建人格更新器成功")
            return service
        except ImportError as e:
            self._logger.error(f"导入人格更新器失败: {e}", exc_info=True)
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
            self.create_intelligent_responder()  # 重新启用智能回复器
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
            self._logger.error(f"服务初始化异常: {e}", exc_info=True)
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
            self._logger.error(f"服务关闭异常: {e}", exc_info=True)
            return False
    
    def get_service_status(self) -> Dict[str, str]:
        """获取所有服务状态"""
        return self._registry.get_service_status()
    
    def clear_cache(self):
        """清理服务缓存"""
        self._service_cache.clear()
        self._logger.info("服务缓存已清理")


# 将内部类移到模块顶层
class QQFilter:
    def __init__(self, target_qq_list):
        self.target_qq_list = target_qq_list or []
        self._logger = logger
    
    def should_collect_message(self, sender_id: str) -> bool:
        if not self.target_qq_list:  # 空列表表示收集所有
            return True
        return sender_id in self.target_qq_list


class MessageFilter:
    def __init__(self, config: PluginConfig, context: Context, llm_client=None, prompts: Any = None):
        self.config = config
        self.context = context
        # llm_client已弃用，不再使用
        self.prompts = prompts  # 保存 prompts
        self._logger = logger
    
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
            
            prompt = self.prompts.MESSAGE_FILTER_SUITABLE_FOR_LEARNING_PROMPT.format(
                current_persona=current_persona,
                message=message
            )
            
            # LLMClient已弃用，返回默认结果
            return False  # 默认认为不适合学习
        except Exception as e:
            self._logger.error(f"LLM 筛选消息失败: {e}", exc_info=True)
            return False # LLM 调用失败，认为不适合


class LearningScheduler:
    def __init__(self, plugin_instance):
        self.plugin = plugin_instance
        self.is_running = False
        self._task = None
        self._logger = logger
    
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
                interval_seconds = self.plugin.plugin_config.learning_interval_hours * 3600 # 使用 plugin_config
                await asyncio.sleep(interval_seconds)
                
                if self.is_running and hasattr(self.plugin, '_perform_learning_cycle'):
                    await self.plugin._perform_learning_cycle()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"学习循环异常: {e}", exc_info=True)
                await asyncio.sleep(60)  # 错误后等待1分钟再重试


class ComponentFactory:
    """组件工厂 - 创建轻量级组件"""
    
    def __init__(self, config: PluginConfig, service_factory: ServiceFactory):
        self.config = config
        self.service_factory = service_factory
        self._logger = logger
        # 添加服务缓存和注册表引用
        self._service_cache = service_factory._service_cache
        self._registry = service_factory._registry
    
    def create_qq_filter(self):
        """创建QQ号过滤器"""
        return QQFilter(self.config.target_qq_list)
    
    def create_message_filter(self, context: Context): # 不再直接接收 llm_client
        """创建消息过滤器"""
        llm_client = None # LLMClient已弃用，不再传递
        prompts = self.service_factory.get_prompts() # 通过 service_factory 获取 prompts
        return MessageFilter(self.config, context, llm_client, prompts) # 传递 llm_client 和 prompts
    
    def create_learning_scheduler(self, plugin_instance):
        """创建学习调度器"""
        return LearningScheduler(plugin_instance)
    
    def create_persona_updater(self, context: Context, backup_manager):
        """创建人格更新器"""
        from ..services.persona_updater import PersonaUpdater as ActualPersonaUpdater # 导入实际的 PersonaUpdater
        llm_client = None # LLMClient已弃用，不再传递
        prompts = self.service_factory.get_prompts() # 获取 prompts
        return ActualPersonaUpdater(self.config, context, backup_manager, llm_client, prompts) # 传递 config, llm_client 和 prompts

    def create_data_analytics_service(self):
        """创建数据分析与可视化服务"""
        cache_key = "data_analytics"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.data_analytics import DataAnalyticsService
            
            service = DataAnalyticsService(
                self.config,
                self.service_factory.create_database_manager()
            )
            self._service_cache[cache_key] = service
            self._registry.register_service("data_analytics", service)
            
            self._logger.info("创建数据分析服务成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入数据分析服务失败: {e}", exc_info=True)
            raise ServiceError(f"创建数据分析服务失败: {str(e)}")

    def create_advanced_learning_service(self):
        """创建高级学习机制服务"""
        cache_key = "advanced_learning"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.advanced_learning import AdvancedLearningService
            
            service = AdvancedLearningService(
                self.config,
                database_manager=self.service_factory.create_database_manager(),
                persona_manager=self.service_factory.create_persona_manager(),
                llm_adapter=self.service_factory.create_framework_llm_adapter()  # 使用框架适配器
            )
            self._service_cache[cache_key] = service
            self._registry.register_service("advanced_learning", service)
            
            self._logger.info("创建高级学习服务成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入高级学习服务失败: {e}", exc_info=True)
            raise ServiceError(f"创建高级学习服务失败: {str(e)}")

    def create_enhanced_interaction_service(self):
        """创建增强交互服务"""
        cache_key = "enhanced_interaction"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.enhanced_interaction import EnhancedInteractionService
            
            service = EnhancedInteractionService(
                self.config,
                database_manager=self.service_factory.create_database_manager(),
                llm_adapter=self.service_factory.create_framework_llm_adapter()  # 使用框架适配器
            )
            self._service_cache[cache_key] = service
            self._registry.register_service("enhanced_interaction", service)
            
            self._logger.info("创建增强交互服务成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入增强交互服务失败: {e}", exc_info=True)
            raise ServiceError(f"创建增强交互服务失败: {str(e)}")

    def create_intelligence_enhancement_service(self):
        """创建智能化提升服务"""
        cache_key = "intelligence_enhancement"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.intelligence_enhancement import IntelligenceEnhancementService
            
            service = IntelligenceEnhancementService(
                self.config,
                database_manager=self.service_factory.create_database_manager(),
                persona_manager=self.service_factory.create_persona_manager(),
                llm_adapter=self.service_factory.create_framework_llm_adapter()  # 使用框架适配器
            )
            self._service_cache[cache_key] = service
            self._registry.register_service("intelligence_enhancement", service)
            
            self._logger.info("创建智能化提升服务成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入智能化提升服务失败: {e}", exc_info=True)
            raise ServiceError(f"创建智能化提升服务失败: {str(e)}")

    def create_affection_manager_service(self):
        """创建好感度管理服务"""
        cache_key = "affection_manager"
        
        if cache_key in self._service_cache:
            return self._service_cache[cache_key]
        
        try:
            from ..services.affection_manager import AffectionManager
            
            service = AffectionManager(
                self.config,
                self.service_factory.create_database_manager(),
                llm_adapter=self.service_factory.create_framework_llm_adapter()  # 使用框架适配器
            )
            self._service_cache[cache_key] = service
            self._registry.register_service("affection_manager", service)
            
            self._logger.info("创建好感度管理服务成功")
            return service
            
        except ImportError as e:
            self._logger.error(f"导入好感度管理服务失败: {e}", exc_info=True)
            raise ServiceError(f"创建好感度管理服务失败: {str(e)}")


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
        self._component_factory = ComponentFactory(config, self._service_factory) # 注入 service_factory
    
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
    
    def get_service(self, service_name: str) -> Any:
        """
        通过名称获取已创建的服务实例。
        此方法委托给 ServiceFactory 的内部缓存。
        """
        if self._service_factory is None:
            raise ServiceError("服务工厂未初始化，无法获取服务")
        
        service = self._service_factory._service_cache.get(service_name)
        if service is None:
            raise ServiceError(f"服务 '{service_name}' 未找到或未初始化。请确保服务已通过 ServiceFactory 的 create_xxx 方法创建。")
        return service

    async def cleanup(self):
        """清理所有工厂"""
        if self._service_factory:
            await self._service_factory.shutdown_all_services()
            self._service_factory.clear_cache()
        
        self._service_factory = None
        self._component_factory = None
