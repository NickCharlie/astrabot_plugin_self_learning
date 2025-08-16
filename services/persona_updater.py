"""
人格更新服务 - 基于AstrBot框架的人格管理
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import copy

from astrbot.api.star import Context
from astrbot.core.provider.provider import Personality
from ..config import PluginConfig
from ..core.interfaces import IPersonaManager, AnalysisResult


class PersonaUpdater(IPersonaManager):
    """
    基于AstrBot框架的人格更新器
    直接操作框架的 curr_personality 属性
    """
    
    def __init__(self, config: PluginConfig, context: Context, backup_manager=None):
        self.config = config
        self.context = context
        self.backup_manager = backup_manager
        self._logger = logging.getLogger(self.__class__.__name__)
        
    async def update_persona(self, style_data: Dict[str, Any]) -> bool:
        """更新当前人格"""
        try:
            # 获取当前提供商
            provider = self.context.get_using_provider()
            if not provider:
                self._logger.error("无法获取当前LLM提供商")
                return False
            
            # 检查是否有当前人格
            if not hasattr(provider, 'curr_personality') or not provider.curr_personality:
                self._logger.error("当前提供商没有设置人格")
                return False
            
            current_persona = provider.curr_personality
            self._logger.info(f"当前人格: {current_persona.get('name', 'unknown')}")
            
            # 创建备份
            if self.backup_manager:
                backup_id = await self.backup_persona(f"自动备份_更新前_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                self._logger.info(f"创建人格备份: {backup_id}")
            
            # 更新人格prompt
            if 'enhanced_prompt' in style_data:
                original_prompt = current_persona.get('prompt', '')
                enhanced_prompt = self._merge_prompts(original_prompt, style_data['enhanced_prompt'])
                current_persona['prompt'] = enhanced_prompt
                self._logger.info(f"人格prompt已更新，长度: {len(enhanced_prompt)}")
            
            # 更新对话风格模仿
            if 'filtered_messages' in style_data:
                await self._update_mood_imitation_dialogs(current_persona, style_data['filtered_messages'])
            
            # 更新其他风格属性
            if 'style_attributes' in style_data:
                await self._apply_style_attributes(current_persona, style_data['style_attributes'])
            
            self._logger.info("人格更新成功")
            return True
            
        except Exception as e:
            self._logger.error(f"人格更新失败: {e}")
            return False
    
    async def backup_persona(self, reason: str) -> int:
        """备份当前人格"""
        try:
            provider = self.context.get_using_provider()
            if not provider or not provider.curr_personality:
                raise ValueError("无法获取当前人格进行备份")
            
            if self.backup_manager:
                return await self.backup_manager.create_backup_before_update("system", reason)
            else:
                # 简单备份到日志
                persona_copy = copy.deepcopy(provider.curr_personality)
                backup_id = int(datetime.now().timestamp())
                self._logger.info(f"人格备份 {backup_id}: {persona_copy}")
                return backup_id
            
        except Exception as e:
            self._logger.error(f"人格备份失败: {e}")
            return -1
    
    async def restore_persona(self, backup_id: int) -> bool:
        """恢复人格"""
        try:
            if self.backup_manager:
                return await self.backup_manager.restore_persona_from_backup(backup_id)
            else:
                self._logger.warning("没有配置备份管理器，无法恢复人格")
                return False
            
        except Exception as e:
            self._logger.error(f"人格恢复失败: {e}")
            return False
    
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
    
    def _merge_prompts(self, original: str, enhancement: str) -> str:
        """合并原始prompt和增强prompt"""
        if not original:
            return enhancement
        
        if not enhancement:
            return original
        
        # 智能合并策略
        if self.config.persona_merge_strategy == "replace":
            return enhancement
        elif self.config.persona_merge_strategy == "append":
            return f"{original}\n\n{enhancement}"
        elif self.config.persona_merge_strategy == "prepend":
            return f"{enhancement}\n\n{original}"
        else:  # smart merge
            return self._smart_merge_prompts(original, enhancement)
    
    def _smart_merge_prompts(self, original: str, enhancement: str) -> str:
        """智能合并prompt"""
        # 检查重叠内容，避免重复
        words_original = set(original.lower().split())
        words_enhancement = set(enhancement.lower().split())
        
        overlap_ratio = len(words_original.intersection(words_enhancement)) / max(len(words_original), 1)
        
        if overlap_ratio > 0.7:  # 高重叠，选择较长的
            return enhancement if len(enhancement) > len(original) else original
        else:  # 低重叠，合并
            return f"{original}\n\n补充风格特征：{enhancement}"
    
    async def _update_mood_imitation_dialogs(self, persona: Personality, filtered_messages: List[Dict[str, Any]]):
        """更新对话风格模仿"""
        try:
            current_dialogs = persona.get('mood_imitation_dialogs', [])
            
            # 从过滤后的消息中提取高质量对话
            new_dialogs = []
            for msg in filtered_messages[-10:]:  # 取最近10条
                message_text = msg.get('message', '').strip()
                if message_text and len(message_text) > self.config.message_min_length:
                    if message_text not in current_dialogs:
                        new_dialogs.append(message_text)
            
            if new_dialogs:
                # 保持对话列表长度合理
                max_dialogs = self.config.max_mood_imitation_dialogs or 20
                all_dialogs = current_dialogs + new_dialogs
                
                if len(all_dialogs) > max_dialogs:
                    # 保留最新的对话
                    all_dialogs = all_dialogs[-max_dialogs:]
                
                persona['mood_imitation_dialogs'] = all_dialogs
                self._logger.info(f"更新对话风格模仿，新增{len(new_dialogs)}条，总计{len(all_dialogs)}条")
            
        except Exception as e:
            self._logger.error(f"更新对话风格模仿失败: {e}")
    
    async def _apply_style_attributes(self, persona: Personality, style_attributes: Dict[str, Any]):
        """应用风格属性"""
        try:
            current_prompt = persona.get('prompt', '')
            
            # 根据风格属性调整prompt
            if 'tone' in style_attributes:
                tone = style_attributes['tone']
                tone_instruction = f"请保持{tone}的语调。"
                if tone_instruction not in current_prompt:
                    current_prompt = f"{current_prompt}\n\n{tone_instruction}"
            
            if 'formality' in style_attributes:
                formality = style_attributes['formality']
                if formality == 'formal':
                    formality_instruction = "请使用正式的表达方式。"
                elif formality == 'casual':
                    formality_instruction = "请使用轻松随意的表达方式。"
                else:
                    formality_instruction = ""
                
                if formality_instruction and formality_instruction not in current_prompt:
                    current_prompt = f"{current_prompt}\n\n{formality_instruction}"
            
            if 'emotion' in style_attributes:
                emotion = style_attributes['emotion']
                if emotion and f"情感倾向：{emotion}" not in current_prompt:
                    current_prompt = f"{current_prompt}\n\n情感倾向：{emotion}"
            
            persona['prompt'] = current_prompt
            self._logger.info("风格属性应用成功")
            
        except Exception as e:
            self._logger.error(f"应用风格属性失败: {e}")
    
    async def analyze_persona_compatibility(self, target_style: Dict[str, Any]) -> AnalysisResult:
        """分析目标风格与当前人格的兼容性"""
        try:
            current_persona = await self.get_current_persona()
            if not current_persona:
                return AnalysisResult(
                    success=False,
                    confidence=0.0,
                    data={},
                    error="无法获取当前人格"
                )
            
            current_prompt = current_persona.get('prompt', '')
            target_attributes = target_style.get('style_attributes', {})
            
            # 简单的兼容性评分
            compatibility_score = 0.8  # 基础分数
            
            # 检查风格冲突
            conflicts = []
            if 'tone' in target_attributes:
                target_tone = target_attributes['tone'].lower()
                if ('严肃' in current_prompt.lower() and target_tone == 'humor') or \
                   ('幽默' in current_prompt.lower() and target_tone == 'serious'):
                    conflicts.append('语调冲突')
                    compatibility_score -= 0.2
            
            return AnalysisResult(
                success=True,
                confidence=compatibility_score,
                data={
                    'compatibility_score': compatibility_score,
                    'conflicts': conflicts,
                    'current_persona_name': current_persona.get('name', 'unknown'),
                    'recommended_action': 'merge' if compatibility_score > 0.6 else 'replace'
                }
            )
            
        except Exception as e:
            self._logger.error(f"人格兼容性分析失败: {e}")
            return AnalysisResult(
                success=False,
                confidence=0.0,
                data={},
                error=str(e)
            )


class PersonaAnalyzer:
    """人格分析器 - 分析人格特征和变化"""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
    
    async def analyze_persona_evolution(self, persona_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析人格演化趋势"""
        if len(persona_history) < 2:
            return {
                'evolution_detected': False,
                'message': '人格历史数据不足'
            }
        
        try:
            # 分析prompt长度变化
            prompt_lengths = [len(p.get('prompt', '')) for p in persona_history]
            length_trend = 'increasing' if prompt_lengths[-1] > prompt_lengths else 'decreasing'
            
            # 分析关键词变化
            all_keywords = []
            for persona in persona_history:
                prompt = persona.get('prompt', '').lower()
                keywords = self._extract_keywords(prompt)
                all_keywords.extend(keywords)
            
            keyword_frequency = {}
            for keyword in all_keywords:
                keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1
            
            most_common_keywords = sorted(keyword_frequency.items(), key=lambda x: x, reverse=True)[:10][1]
            
            return {
                'evolution_detected': True,
                'prompt_length_trend': length_trend,
                'most_common_keywords': most_common_keywords,
                'total_versions': len(persona_history),
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self._logger.error(f"人格演化分析失败: {e}")
            return {
                'evolution_detected': False,
                'error': str(e)
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单的关键词提取
        words = text.split()
        keywords = []
        
        important_words = ['友好', '专业', '幽默', '严肃', '活泼', '温和', '耐心', '热情']
        
        for word in words:
            if any(important in word for important in important_words):
                keywords.append(word)
        
        return keywords
