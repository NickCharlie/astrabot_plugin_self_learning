"""
AstrBot 框架 LLM 适配器
用于替换自定义 LLMClient，直接使用 AstrBot 框架的 Provider 系统
"""
import asyncio
from typing import Optional, List, Dict, Any
from astrbot.api import logger
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.entities import LLMResponse

class FrameworkLLMAdapter:
    """AstrBot框架LLM适配器，用于替换自定义LLMClient"""
    
    def __init__(self, context):
        self.context = context
        self.filter_provider: Optional[Provider] = None
        self.refine_provider: Optional[Provider] = None  
        self.reinforce_provider: Optional[Provider] = None
        
    def initialize_providers(self, config):
        """根据配置初始化Provider"""
        if config.filter_provider_id:
            self.filter_provider = self.context.get_provider_by_id(config.filter_provider_id)
            if not self.filter_provider:
                logger.warning(f"找不到筛选Provider: {config.filter_provider_id}")
            else:
                logger.info(f"筛选Provider已配置: {config.filter_provider_id}")
                
        if config.refine_provider_id:
            self.refine_provider = self.context.get_provider_by_id(config.refine_provider_id)
            if not self.refine_provider:
                logger.warning(f"找不到提炼Provider: {config.refine_provider_id}")
            else:
                logger.info(f"提炼Provider已配置: {config.refine_provider_id}")
                
        if config.reinforce_provider_id:
            self.reinforce_provider = self.context.get_provider_by_id(config.reinforce_provider_id)
            if not self.reinforce_provider:
                logger.warning(f"找不到强化Provider: {config.reinforce_provider_id}")
            else:
                logger.info(f"强化Provider已配置: {config.reinforce_provider_id}")
    
    async def filter_chat_completion(
        self,
        prompt: str,
        contexts: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """使用筛选模型进行对话补全"""
        if not self.filter_provider:
            logger.error("筛选Provider未配置")
            return None
            
        try:
            logger.debug(f"调用筛选Provider: {self.filter_provider.meta().id}")
            response = await self.filter_provider.text_chat(
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt,
                **kwargs
            )
            return response.completion_text if response else None
        except Exception as e:
            logger.error(f"筛选模型调用失败: {e}")
            return None
    
    async def refine_chat_completion(
        self,
        prompt: str,
        contexts: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """使用提炼模型进行对话补全"""
        if not self.refine_provider:
            logger.error("提炼Provider未配置")
            return None
            
        try:
            logger.debug(f"调用提炼Provider: {self.refine_provider.meta().id}")
            response = await self.refine_provider.text_chat(
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt,
                **kwargs
            )
            return response.completion_text if response else None
        except Exception as e:
            logger.error(f"提炼模型调用失败: {e}")
            return None
    
    async def reinforce_chat_completion(
        self,
        prompt: str,
        contexts: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """使用强化模型进行对话补全"""
        if not self.reinforce_provider:
            logger.error("强化Provider未配置")
            return None
            
        try:
            logger.debug(f"调用强化Provider: {self.reinforce_provider.meta().id}")
            response = await self.reinforce_provider.text_chat(
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt,
                **kwargs
            )
            return response.completion_text if response else None
        except Exception as e:
            logger.error(f"强化模型调用失败: {e}")
            return None

    def has_filter_provider(self) -> bool:
        """检查是否有筛选Provider"""
        return self.filter_provider is not None
    
    def has_refine_provider(self) -> bool:
        """检查是否有提炼Provider"""
        return self.refine_provider is not None
    
    def has_reinforce_provider(self) -> bool:
        """检查是否有强化Provider"""
        return self.reinforce_provider is not None

    def get_provider_info(self) -> Dict[str, str]:
        """获取Provider信息"""
        info = {}
        if self.filter_provider:
            info['filter'] = f"{self.filter_provider.meta().id} ({self.filter_provider.meta().model})"
        if self.refine_provider:
            info['refine'] = f"{self.refine_provider.meta().id} ({self.refine_provider.meta().model})"
        if self.reinforce_provider:
            info['reinforce'] = f"{self.reinforce_provider.meta().id} ({self.reinforce_provider.meta().model})"
        return info