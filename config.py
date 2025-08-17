"""
自学习插件配置管理
"""
import os
from typing import List, Optional
from dataclasses import dataclass
from astrbot.core.utils.astrbot_path import get_astrbot_data_path


@dataclass
class PluginConfig:
    """插件配置类"""
    
    # 基础开关
    enable_message_capture: bool = True
    enable_auto_learning: bool = True  
    enable_realtime_learning: bool = False
    enable_web_interface: bool = False
    
    # QQ号设置
    target_qq_list: List[str] = None
    
    # 模型配置
    filter_model_name: str = "gpt-4o-mini"  # 筛选模型（弱模型）
    refine_model_name: str = "gpt-4o"       # 提炼模型（强模型）
    reinforce_model_name: str = "gpt-4o"    # 强化模型（用于强化学习）

    # LLM 提供商 ID
    filter_provider_id: Optional[str] = None  # 筛选模型使用的提供商ID
    refine_provider_id: Optional[str] = None  # 提炼模型使用的提供商ID
    reinforce_provider_id: Optional[str] = None # 强化模型使用的提供商ID

    filter_api_url: Optional[str] = None # 筛选模型API URL
    filter_api_key: Optional[str] = None # 筛选模型API Key
    refine_api_url: Optional[str] = None # 提炼模型API URL
    refine_api_key: Optional[str] = None # 提炼模型API Key
    reinforce_api_url: Optional[str] = None # 强化模型API URL
    reinforce_api_key: Optional[str] = None # 强化模型API Key
    
    # 当前人格设置
    current_persona_name: str = "default"
    
    # 学习参数
    learning_interval_hours: int = 6        # 学习间隔（小时）
    min_messages_for_learning: int = 50     # 最少消息数量才开始学习
    max_messages_per_batch: int = 200       # 每批处理的最大消息数量
    
    # 筛选参数
    message_min_length: int = 5             # 消息最小长度
    message_max_length: int = 500           # 消息最大长度
    confidence_threshold: float = 0.7       # 筛选置信度阈值
    
    # 风格分析参数
    style_analysis_batch_size: int = 100    # 风格分析批次大小
    style_update_threshold: float = 0.8     # 风格更新阈值
    
    # 机器学习设置
    enable_ml_analysis: bool = True          # 启用ML分析
    max_ml_sample_size: int = 100           # ML样本最大数量
    ml_cache_timeout_hours: int = 1         # ML缓存超时
    
    # 智能回复设置
    enable_intelligent_reply: bool = False   # 启用智能回复
    reply_probability: float = 0.1          # 回复概率
    context_window_size: int = 5            # 上下文窗口大小
    intelligent_reply_keywords: List[str] = None # 智能回复关键词
    
    # 人格备份设置
    auto_backup_enabled: bool = True        # 启用自动备份
    backup_interval_hours: int = 24         # 备份间隔
    max_backups_per_group: int = 10         # 每群最大备份数
    
    # 高级设置
    debug_mode: bool = False                # 调试模式
    save_raw_messages: bool = True          # 保存原始消息
    auto_backup_interval_days: int = 7      # 自动备份间隔
    
    # PersonaUpdater配置
    persona_merge_strategy: str = "smart"   # 人格合并策略: "replace", "append", "prepend", "smart"
    max_mood_imitation_dialogs: int = 20    # 最大对话风格模仿数量
    enable_persona_evolution: bool = True   # 启用人格演化跟踪
    persona_compatibility_threshold: float = 0.6  # 人格兼容性阈值
    
    # 存储路径
    data_dir: str = ""
    messages_db_path: str = ""
    learning_log_path: str = ""
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.data_dir:
            self.data_dir = os.path.join(get_astrbot_data_path(), "self_learning")
            
        if not self.messages_db_path:
            self.messages_db_path = os.path.join(self.data_dir, "messages.db")
            
        if not self.learning_log_path:
            self.learning_log_path = os.path.join(self.data_dir, "learning.log")
            
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化target_qq_list为空列表
        if self.target_qq_list is None:
            self.target_qq_list = []
        
        # 初始化intelligent_reply_keywords为空列表
        if self.intelligent_reply_keywords is None:
            self.intelligent_reply_keywords = ['bot', 'ai', '人工智能', '机器人', '助手']

    @classmethod
    def create_from_config(cls, config: dict) -> 'PluginConfig':
        """从AstrBot配置创建插件配置"""
        
        # 从配置中提取插件相关设置
        plugin_settings = config.get('self_learning_settings', {})
        
        return cls(
            enable_message_capture=plugin_settings.get('enable_message_capture', True),
            enable_auto_learning=plugin_settings.get('enable_auto_learning', True),
            enable_realtime_learning=plugin_settings.get('enable_realtime_learning', False),
            enable_web_interface=plugin_settings.get('enable_web_interface', False),
            
            target_qq_list=plugin_settings.get('target_qq_list', []),
            
            filter_model_name=plugin_settings.get('filter_model_name', 'gpt-4o-mini'),
            refine_model_name=plugin_settings.get('refine_model_name', 'gpt-4o'),
            reinforce_model_name=plugin_settings.get('reinforce_model_name', 'gpt-4o'),

            filter_provider_id=plugin_settings.get('filter_provider_id', None),
            refine_provider_id=plugin_settings.get('refine_provider_id', None),
            reinforce_provider_id=plugin_settings.get('reinforce_provider_id', None),

            filter_api_url=plugin_settings.get('filter_api_url', None),
            filter_api_key=plugin_settings.get('filter_api_key', None),
            refine_api_url=plugin_settings.get('refine_api_url', None),
            refine_api_key=plugin_settings.get('refine_api_key', None),
            reinforce_api_url=plugin_settings.get('reinforce_api_url', None),
            reinforce_api_key=plugin_settings.get('reinforce_api_key', None),
            
            current_persona_name=plugin_settings.get('current_persona_name', 'default'),
            
            learning_interval_hours=plugin_settings.get('learning_interval_hours', 6),
            min_messages_for_learning=plugin_settings.get('min_messages_for_learning', 50),
            max_messages_per_batch=plugin_settings.get('max_messages_per_batch', 200),
            
            message_min_length=plugin_settings.get('message_min_length', 5),
            message_max_length=plugin_settings.get('message_max_length', 500),
            confidence_threshold=plugin_settings.get('confidence_threshold', 0.7),
            
            style_analysis_batch_size=plugin_settings.get('style_analysis_batch_size', 100),
            style_update_threshold=plugin_settings.get('style_update_threshold', 0.8)
        )

    @classmethod
    def create_default(cls) -> 'PluginConfig':
        """创建默认配置"""
        return cls()

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'enable_message_capture': self.enable_message_capture,
            'enable_auto_learning': self.enable_auto_learning,
            'enable_realtime_learning': self.enable_realtime_learning,
            'enable_web_interface': self.enable_web_interface,
            'target_qq_list': self.target_qq_list,
            'filter_model_name': self.filter_model_name,
            'refine_model_name': self.refine_model_name,
            'reinforce_model_name': self.reinforce_model_name,
            'filter_provider_id': self.filter_provider_id,
            'refine_provider_id': self.refine_provider_id,
            'reinforce_provider_id': self.reinforce_provider_id,
            'filter_api_url': self.filter_api_url,
            'filter_api_key': self.filter_api_key,
            'refine_api_url': self.refine_api_url,
            'refine_api_key': self.refine_api_key,
            'reinforce_api_url': self.reinforce_api_url,
            'reinforce_api_key': self.reinforce_api_key,
            'current_persona_name': self.current_persona_name,
            'learning_interval_hours': self.learning_interval_hours,
            'min_messages_for_learning': self.min_messages_for_learning,
            'max_messages_per_batch': self.max_messages_per_batch,
            'message_min_length': self.message_min_length,
            'message_max_length': self.message_max_length,
            'confidence_threshold': self.confidence_threshold,
            'style_analysis_batch_size': self.style_analysis_batch_size,
            'style_update_threshold': self.style_update_threshold
        }

    def validate(self) -> List[str]:
        """验证配置有效性，返回错误信息列表"""
        errors = []
        
        if self.learning_interval_hours <= 0:
            errors.append("学习间隔必须大于0小时")
            
        if self.min_messages_for_learning <= 0:
            errors.append("最少学习消息数量必须大于0")
            
        if self.max_messages_per_batch <= 0:
            errors.append("每批最大消息数量必须大于0")
            
        if self.message_min_length >= self.message_max_length:
            errors.append("消息最小长度必须小于最大长度")
            
        if not 0 <= self.confidence_threshold <= 1:
            errors.append("置信度阈值必须在0-1之间")
            
        if not 0 <= self.style_update_threshold <= 1:
            errors.append("风格更新阈值必须在0-1之间")
            
        if not self.filter_model_name:
            errors.append("筛选模型名称不能为空")
            
        if not self.refine_model_name:
            errors.append("提炼模型名称不能为空")
        
        if not self.reinforce_model_name:
            errors.append("强化模型名称不能为空")
            
        return errors
