import logging
from typing import Dict, Any, Optional, List

from astrbot.api.star import Context
from ..config import PluginConfig
from ..core.interfaces import IPersonaManager, IPersonaUpdater, IPersonaBackupManager, ServiceLifecycle, MessageData
from ..exceptions import SelfLearningError # 导入 SelfLearningError

class PersonaManagerService(IPersonaManager):
    """
    人格管理器服务，负责协调人格的更新、备份和恢复。
    实现 IPersonaManager 接口。
    """
    def __init__(self, config: PluginConfig, context: Context,
                 persona_updater: IPersonaUpdater, persona_backup_manager: IPersonaBackupManager):
        self.config = config
        self.context = context
        self._logger = logging.getLogger(self.__class__.__name__)
        self._persona_updater = persona_updater
        self._persona_backup_manager = persona_backup_manager
        self._status = ServiceLifecycle.CREATED

    @property
    def status(self) -> ServiceLifecycle:
        return self._status

    async def start(self) -> bool:
        self._status = ServiceLifecycle.RUNNING
        self._logger.info("PersonaManagerService started.")
        return True

    async def stop(self) -> bool:
        self._status = ServiceLifecycle.STOPPED
        self._logger.info("PersonaManagerService stopped.")
        return True

    async def restart(self) -> bool:
        await self.stop()
        return await self.start()

    async def health_check(self) -> bool:
        return self._status == ServiceLifecycle.RUNNING

    async def update_persona(self, style_data: Dict[str, Any], messages: List[MessageData]) -> bool:
        """
        更新人格。
        此方法将委托给 PersonaUpdater。
        """
        try:
            self._logger.info("PersonaManagerService: Updating persona...")
            # 在更新前创建备份
            backup_id = await self._persona_backup_manager.create_backup_before_update(
                "default",  # 假设 group_id 为 "default"
                f"Style update initiated by PersonaManagerService"
            )
            self._logger.info(f"PersonaManagerService: Created persona backup: {backup_id}")

            # 调用 PersonaUpdater 的方法进行实际更新
            update_success = await self._persona_updater.update_persona_with_style(style_data, messages)
            
            if update_success:
                self._logger.info("PersonaManagerService: Persona updated successfully.")
            else:
                self._logger.warning("PersonaManagerService: Persona update failed via PersonaUpdater.")
            
            return update_success
            
        except Exception as e:
            self._logger.error(f"PersonaManagerService: Failed to update persona: {e}")
            raise SelfLearningError(f"人格更新失败: {str(e)}") from e

    async def backup_persona(self, reason: str) -> int:
        """
        备份人格。
        此方法将委托给 PersonaBackupManager。
        """
        try:
            self._logger.info(f"PersonaManagerService: Backing up persona with reason: {reason}")
            # 假设 group_id 为 "default"
            backup_id = await self._persona_backup_manager.create_backup_before_update("default", reason)
            self._logger.info(f"PersonaManagerService: Persona backup created with ID: {backup_id}")
            return backup_id
        except Exception as e:
            self._logger.error(f"PersonaManagerService: Failed to backup persona: {e}")
            raise SelfLearningError(f"人格备份失败: {str(e)}") from e

    async def restore_persona(self, backup_id: int) -> bool:
        """
        恢复人格。
        此方法将委托给 PersonaBackupManager。
        """
        try:
            self._logger.info(f"PersonaManagerService: Restoring persona from backup ID: {backup_id}")
            # 假设 group_id 为 "default"
            restore_success = await self._persona_backup_manager.restore_persona("default", backup_id)
            if restore_success:
                self._logger.info(f"PersonaManagerService: Persona restored successfully from backup ID: {backup_id}")
            else:
                self._logger.warning(f"PersonaManagerService: Failed to restore persona from backup ID: {backup_id}")
            return restore_success
        except Exception as e:
            self._logger.error(f"PersonaManagerService: Failed to restore persona: {e}")
            raise SelfLearningError(f"人格恢复失败: {str(e)}") from e

    async def get_current_persona_description(self) -> Optional[str]:
        """获取当前人格的描述"""
        try:
            provider = self.context.get_using_provider()
            if provider and provider.curr_personality:
                return provider.curr_personality.get('prompt', '')
            return None
        except Exception as e:
            self._logger.error(f"获取当前人格描述失败: {e}")
            return None

    async def get_current_persona(self) -> Optional[Dict[str, Any]]:
        """获取当前人格信息"""
        try:
            provider = self.context.get_using_provider()
            if provider and provider.curr_personality:
                return dict(provider.curr_personality)
            return None
        except Exception as e:
            self._logger.error(f"获取当前人格失败: {e}")
            return None
