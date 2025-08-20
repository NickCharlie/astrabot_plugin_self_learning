"""
风格分析服务 - 使用强模型深度分析对话风格并提炼特征
"""
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.star import Context

from ..config import PluginConfig
from ..exceptions import StyleAnalysisError, ModelAccessError
from ..core.llm_client import LLMClient # 导入 LLMClient
from .database_manager import DatabaseManager # 导入 DatabaseManager


@dataclass
class StyleProfile:
    """风格档案"""
    vocabulary_richness: float = 0.0      # 词汇丰富度
    sentence_complexity: float = 0.0      # 句式复杂度
    emotional_expression: float = 0.0     # 情感表达度
    interaction_tendency: float = 0.0     # 互动倾向
    topic_diversity: float = 0.0          # 话题多样性
    formality_level: float = 0.0          # 正式程度
    creativity_score: float = 0.0         # 创造性得分


@dataclass
class StyleEvolution:
    """风格演化记录"""
    timestamp: str
    old_profile: StyleProfile
    new_profile: StyleProfile
    evolution_vector: Dict[str, float]
    significance: float


class StyleAnalyzerService:
    """风格分析服务"""
    
    def __init__(self, config: PluginConfig, context: Context, database_manager: DatabaseManager, 
                 refine_llm_client: Optional[LLMClient], prompts: Any): # 添加 refine_llm_client 和 prompts
        self.config = config
        self.context = context
        self.db_manager = database_manager # 注入 DatabaseManager 实例
        self.refine_llm_client = refine_llm_client # 保存 LLMClient 实例
        self.prompts = prompts # 保存 prompts
        
        # 风格演化历史
        self.style_evolution_history: List[StyleEvolution] = []
        
        # 当前基准风格档案
        self.baseline_style: Optional[StyleProfile] = None
        
        logger.info("风格分析服务初始化完成")

    async def start(self):
        """服务启动时加载基准风格档案"""
        # 假设每个群组有独立的风格档案，这里需要一个 group_id
        # 为了简化，暂时假设加载一个默认的或全局的风格档案
        # 实际应用中，可能需要根据当前处理的群组ID来加载
        default_group_id = "global_style" # 或者从配置中获取
        loaded_profile_data = await self.db_manager.load_style_profile(default_group_id, "baseline_style_profile")
        if loaded_profile_data:
            self.baseline_style = StyleProfile(**loaded_profile_data)
            logger.info("已从数据库加载基准风格档案。")
        else:
            logger.info("未找到基准风格档案，将从零开始。")

    async def analyze_conversation_style(self, group_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析对话风格，使用强模型进行深度分析"""
        try:
            if not messages:
                return {"error": "没有消息数据"}
            
            # 获取强模型LLM客户端
            llm_client = self._get_refine_model_client()
            if not llm_client:
                return {"error": "提炼模型LLM客户端未初始化，无法进行风格分析。"}
            
            # 准备分析数据
            message_texts = [msg.get('message', '') for msg in messages]
            combined_text = '\n'.join(message_texts[:50])  # 限制长度避免token超限
            
            # 生成风格分析报告
            style_analysis = await self._generate_style_analysis(combined_text, llm_client)
            
            # 提取数值化特征
            style_profile = await self._extract_style_profile(combined_text, llm_client)
            
            # 检测风格变化
            style_evolution = None
            if self.baseline_style:
                style_evolution = self._detect_style_evolution(self.baseline_style, style_profile)
            
            # 更新基准风格并持久化
            self.baseline_style = style_profile
            await self.db_manager.save_style_profile(group_id, {"profile_name": "baseline_style_profile", **self.baseline_style.__dict__})
            
            return {
                'style_analysis': style_analysis,
                'style_profile': self.baseline_style.__dict__, # 返回更新后的基准风格
                'style_evolution': style_evolution.__dict__ if style_evolution else None,
                'message_count': len(messages),
                'analysis_timestamp': datetime.now().isoformat(),
                'confidence': await self._calculate_analysis_confidence(messages)
            }
            
        except Exception as e:
            logger.error(f"对话风格分析失败: {e}")
            raise StyleAnalysisError(f"风格分析失败: {str(e)}")

    async def _generate_style_analysis(self, text: str, llm_client: LLMClient) -> Dict[str, Any]:
        """生成详细的风格分析报告"""
        try:
            prompt = self.prompts.STYLE_ANALYZER_GENERATE_STYLE_ANALYSIS_PROMPT.format(
                text=text
            )
            
            response = await llm_client.chat_completion(prompt=prompt)
            
            if response and response.text():
                try:
                    analysis = json.loads(response.text().strip())
                    return analysis
                except json.JSONDecodeError:
                    return {"error": "JSON解析失败", "raw_response": response.text()}
            return {"error": "LLM响应为空"}
                
        except Exception as e:
            logger.error(f"风格分析生成失败: {e}")
            return {"error": str(e)}

    async def _extract_style_profile(self, text: str, llm_client: LLMClient) -> StyleProfile:
        """提取数值化的风格档案"""
        try:
            prompt = self.prompts.STYLE_ANALYZER_EXTRACT_STYLE_PROFILE_PROMPT.format(
                text=text
            )
            
            response = await llm_client.chat_completion(prompt=prompt)
            
            if response and response.text():
                try:
                    scores = json.loads(response.text().strip())
                    return StyleProfile(**scores)
                except (json.JSONDecodeError, TypeError):
                    # 返回默认值
                    return StyleProfile()
            return StyleProfile() # LLM响应为空，返回默认值
                
        except Exception as e:
            logger.warning(f"风格档案提取失败: {e}")
            return StyleProfile()

    def _detect_style_evolution(self, old_style: StyleProfile, new_style: StyleProfile) -> StyleEvolution:
        """检测风格演化"""
        evolution_vector = {
            'vocabulary_richness': new_style.vocabulary_richness - old_style.vocabulary_richness,
            'sentence_complexity': new_style.sentence_complexity - old_style.sentence_complexity,
            'emotional_expression': new_style.emotional_expression - old_style.emotional_expression,
            'interaction_tendency': new_style.interaction_tendency - old_style.interaction_tendency,
            'topic_diversity': new_style.topic_diversity - old_style.topic_diversity,
            'formality_level': new_style.formality_level - old_style.formality_level,
            'creativity_score': new_style.creativity_score - old_style.creativity_score
        }
        
        # 计算变化显著性
        significance = sum(abs(v) for v in evolution_vector.values()) / len(evolution_vector)
        
        evolution = StyleEvolution(
            timestamp=datetime.now().isoformat(),
            old_profile=old_style,
            new_profile=new_style,
            evolution_vector=evolution_vector,
            significance=significance
        )
        
        # 存储演化记录
        self.style_evolution_history.append(evolution)
        
        # 保持最近20条记录
        if len(self.style_evolution_history) > 20:
            self.style_evolution_history = self.style_evolution_history[-20:]
        
        return evolution

    async def _calculate_analysis_confidence(self, messages: List[Dict[str, Any]]) -> float:
        """计算分析置信度"""
        confidence = 0.5  # 基础置信度
        
        # 消息数量影响
        message_count = len(messages)
        if message_count >= 100:
            confidence += 0.3
        elif message_count >= 50:
            confidence += 0.2
        elif message_count >= 20:
            confidence += 0.1
        
        # 消息内容质量影响
        total_chars = sum(len(msg.get('message', '')) for msg in messages)
        avg_length = total_chars / max(message_count, 1)
        
        if avg_length >= 50:
            confidence += 0.2
        elif avg_length >= 20:
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _get_refine_model_client(self) -> Optional[LLMClient]:
        """获取提炼模型LLM客户端"""
        if not self.refine_llm_client:
            logger.warning("提炼模型LLM客户端未初始化。")
        return self.refine_llm_client

    async def get_style_trends(self) -> Dict[str, Any]:
        """获取风格趋势分析"""
        if not self.style_evolution_history:
            return {"error": "暂无风格演化数据"}
        
        # 分析最近的风格变化趋势
        recent_evolutions = self.style_evolution_history[-10:]
        
        trends = {}
        for dimension in ['vocabulary_richness', 'sentence_complexity', 'emotional_expression',
                         'interaction_tendency', 'topic_diversity', 'formality_level', 'creativity_score']:
            values = [evo.evolution_vector.get(dimension, 0) for evo in recent_evolutions]
            trends[dimension] = {
                'trend': 'increasing' if sum(values) > 0 else 'decreasing' if sum(values) < 0 else 'stable',
                'average_change': sum(values) / len(values),
                'volatility': sum(abs(v - sum(values)/len(values)) for v in values) / len(values)
            }
        
        return {
            'trends': trends,
            'overall_stability': 1.0 - (sum(evo.significance for evo in recent_evolutions) / len(recent_evolutions)),
            'evolution_count': len(self.style_evolution_history),
            'analysis_period': {
                'start': self.style_evolution_history.timestamp if self.style_evolution_history else None,
                'end': self.style_evolution_history[-1].timestamp if self.style_evolution_history else None
            }
        }

    async def generate_style_recommendations(self, target_persona: str) -> Dict[str, Any]:
        """生成风格优化建议"""
        if not self.baseline_style:
            return {"error": "暂无基准风格数据"}
        
        try:
            llm_client = self._get_refine_model_client()
            if not llm_client:
                return {"error": "提炼模型LLM客户端未初始化，无法生成风格建议。"}
            
            current_style_data = self.baseline_style.__dict__
            
            prompt = self.prompts.STYLE_ANALYZER_GENERATE_STYLE_RECOMMENDATIONS_PROMPT.format(
                current_style_data=json.dumps(current_style_data, ensure_ascii=False, indent=2),
                target_persona=target_persona
            )
            
            response = await llm_client.chat_completion(prompt=prompt)
            
            if response and response.text():
                try:
                    recommendations = json.loads(response.text().strip())
                    return recommendations
                except json.JSONDecodeError:
                    return {"error": "建议解析失败", "raw_response": response.text()}
            return {"error": "LLM响应为空"}
                
        except Exception as e:
            logger.error(f"风格建议生成失败: {e}")
            return {"error": str(e)}
