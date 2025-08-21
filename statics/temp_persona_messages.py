"""
临时人格更新消息常量
"""

class TemporaryPersonaMessages:
    """临时人格更新相关消息"""
    # 备份相关
    BACKUP_CREATED = "人格备份已创建: {backup_name}"
    BACKUP_FILE_CREATED = "人格文件备份已保存: {file_path}"
    BACKUP_FAILED = "人格备份失败: {error}"
    BACKUP_RESTORE_SUCCESS = "人格备份恢复成功: {backup_name}"
    BACKUP_RESTORE_FAILED = "人格备份恢复失败: {error}"
    
    # 临时更新相关
    TEMP_PERSONA_APPLIED = "临时人格已应用: {persona_name} (有效期: {duration}分钟)"
    TEMP_PERSONA_EXPIRED = "临时人格已过期，已恢复原始人格: {original_name}"
    TEMP_PERSONA_REMOVED = "临时人格已手动移除，已恢复原始人格: {original_name}"
    TEMP_PERSONA_CREATE_FAILED = "创建临时人格失败: {error}"
    TEMP_PERSONA_NOT_FOUND = "未找到活动的临时人格"
    
    # 特征学习相关
    FEATURE_LEARNED = "特征学习完成: 已学习 {count} 个新特征"
    DIALOG_LEARNED = "对话学习完成: 已学习 {count} 条对话样本"
    LEARNING_APPLIED = "学习结果已应用到临时人格"
    
    # 错误消息
    ERROR_NO_ORIGINAL_PERSONA = "无法获取原始人格信息"
    ERROR_BACKUP_DIRECTORY_CREATE = "创建备份目录失败: {error}"
    ERROR_TEMP_PERSONA_CONFLICT = "存在冲突的临时人格，请先移除"
    
    # 日志消息
    LOG_BACKUP_CREATED = "为群组 {group_id} 创建人格备份: {backup_name}"
    LOG_TEMP_PERSONA_STARTED = "群组 {group_id} 启动临时人格: {persona_name}"
    LOG_TEMP_PERSONA_EXPIRED = "群组 {group_id} 临时人格已过期: {persona_name}"