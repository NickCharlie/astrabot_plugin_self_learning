"""
插件核心模块
"""

from .factory import ServiceFactory
from .patterns import EventBus, ServiceRegistry, AsyncServiceBase, LearningContext, LearningContextBuilder, StrategyFactory, ConfigurationManager, MetricsCollector
from .interfaces import (
    IMessageCollector, IMessageFilter, IStyleAnalyzer, ILearningStrategy, 
    IQualityMonitor, IPersonaManager, IPersonaUpdater, IPersonaBackupManager, 
    IDataStorage, IObserver, IEventPublisher, IServiceFactory, IAsyncService, 
    IMLAnalyzer, IIntelligentResponder, ServiceLifecycle, MessageData, 
    AnalysisResult, LearningStrategyType, AnalysisType, EventType, 
    PluginError, ServiceError, AnalysisError, ConfigurationError, DataStorageError
)

__all__ = [
    'ServiceFactory',
    'EventBus',
    'ServiceRegistry',
    'AsyncServiceBase',
    'LearningContext',
    'LearningContextBuilder',
    'StrategyFactory',
    'ConfigurationManager',
    'MetricsCollector',
    'IMessageCollector',
    'IMessageFilter',
    'IStyleAnalyzer',
    'ILearningStrategy',
    'IQualityMonitor',
    'IPersonaManager',
    'IPersonaUpdater',
    'IPersonaBackupManager',
    'IDataStorage',
    'IObserver',
    'IEventPublisher',
    'IServiceFactory',
    'IAsyncService',
    'IMLAnalyzer',
    'IIntelligentResponder',
    'ServiceLifecycle',
    'MessageData',
    'AnalysisResult',
    'LearningStrategyType',
    'AnalysisType',
    'EventType',
    'PluginError',
    'ServiceError',
    'AnalysisError',
    'ConfigurationError',
    'DataStorageError'
]
