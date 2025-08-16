"""
轻量级机器学习分析器 - 使用简单的ML算法进行数据分析
"""
import numpy as np
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from astrbot.api import logger

from ..config import PluginConfig
from ..exceptions import AnalysisError


class LightweightMLAnalyzer:
    """轻量级机器学习分析器 - 避免大规模数据分析"""
    
    def __init__(self, config: PluginConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        
        # 设置分析限制以节省资源
        self.max_sample_size = 100  # 最大样本数量
        self.max_features = 50      # 最大特征数量
        self.analysis_cache = {}    # 分析结果缓存
        self.cache_timeout = 3600   # 缓存1小时
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn未安装，将使用基础统计分析")
        
        logger.info("轻量级ML分析器初始化完成")

    async def analyze_user_behavior_pattern(self, group_id: str, user_id: str) -> Dict[str, Any]:
        """分析用户行为模式"""
        try:
            # 检查缓存
            cache_key = f"behavior_{group_id}_{user_id}"
            if self._check_cache(cache_key):
                return self.analysis_cache[cache_key]['data']
            
            # 获取用户最近消息（限制数量）
            messages = await self._get_user_messages(group_id, user_id, limit=self.max_sample_size)
            
            if not messages:
                return {}
            
            # 基础统计分析
            pattern = {
                'message_count': len(messages),
                'avg_message_length': np.mean([len(msg['message']) for msg in messages]),
                'activity_hours': self._analyze_activity_hours(messages),
                'message_frequency': self._analyze_message_frequency(messages),
                'interaction_patterns': await self._analyze_interaction_patterns(group_id, user_id, messages)
            }
            
            # 如果有sklearn，进行文本聚类
            if SKLEARN_AVAILABLE and len(messages) >= 5:
                pattern['topic_clusters'] = self._analyze_topic_clusters(messages)
            
            # 缓存结果
            self._cache_result(cache_key, pattern)
            
            return pattern
            
        except Exception as e:
            logger.error(f"分析用户行为模式失败: {e}")
            raise AnalysisError(f"分析用户行为模式失败: {str(e)}")

    async def _get_user_messages(self, group_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """获取用户消息（限制数量）"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT message, timestamp, sender_name
                FROM raw_messages 
                WHERE sender_id = ? AND timestamp > ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, time.time() - 86400 * 7, limit))  # 最近7天
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'message': row,
                    'timestamp': row[1],
                    'sender_name': row
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"获取用户消息失败: {e}")
            return []

    def _analyze_activity_hours(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """分析活动时间模式"""
        if not messages:
            return {}
        
        hour_counts = defaultdict(int)
        for msg in messages:
            hour = datetime.fromtimestamp(msg['timestamp']).hour
            hour_counts[hour] += 1
        
        total_messages = len(messages)
        hour_distribution = {
            str(hour): count / total_messages 
            for hour, count in hour_counts.items()
        }
        
        # 确定最活跃时段
        most_active_hour = max(hour_counts.items(), key=lambda x: x)[1]
        
        return {
            'distribution': hour_distribution,
            'most_active_hour': most_active_hour,
            'activity_variance': np.var(list(hour_counts.values()))
        }

    def _analyze_message_frequency(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """分析消息频率模式"""
        if len(messages) < 2:
            return {}
        
        # 计算消息间隔
        intervals = []
        sorted_messages = sorted(messages, key=lambda x: x['timestamp'])
        
        for i in range(1, len(sorted_messages)):
            interval = sorted_messages[i]['timestamp'] - sorted_messages[i-1]['timestamp']
            intervals.append(interval / 60)  # 转换为分钟
        
        if not intervals:
            return {}
        
        return {
            'avg_interval_minutes': np.mean(intervals),
            'interval_std': np.std(intervals),
            'burst_tendency': len([x for x in intervals if x < 5]) / len(intervals)  # 5分钟内连续消息比例
        }

    async def _analyze_interaction_patterns(self, group_id: str, user_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析互动模式"""
        try:
            # 分析@消息和回复
            mention_count = len([msg for msg in messages if '@' in msg['message']])
            question_count = len([msg for msg in messages if '?' in msg['message'] or '？' in msg['message']])
            
            # 获取社交关系强度
            social_relations = await self.db_manager.load_social_graph(group_id)
            user_relations = [rel for rel in social_relations if rel['from_user'] == user_id or rel['to_user'] == user_id]
            
            return {
                'mention_ratio': mention_count / max(len(messages), 1),
                'question_ratio': question_count / max(len(messages), 1),
                'social_connections': len(user_relations),
                'avg_relation_strength': np.mean([rel['strength'] for rel in user_relations]) if user_relations else 0.0
            }
            
        except Exception as e:
            logger.error(f"分析互动模式失败: {e}")
            return {}

    def _analyze_topic_clusters(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用TF-IDF和K-means进行话题聚类"""
        if not SKLEARN_AVAILABLE or len(messages) < 3:
            return {}
        
        try:
            # 提取消息文本
            texts = [msg['message'] for msg in messages if len(msg['message']) > 5]
            
            if len(texts) < 3:
                return {}
            
            # TF-IDF向量化（限制特征数量）
            vectorizer = TfidfVectorizer(
                max_features=min(self.max_features, len(texts) * 2),
                stop_words=None,  # 不使用停用词以节省内存
                ngram_range=(1, 1)  # 只使用单词
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # K-means聚类（限制簇数量）
            n_clusters = min(3, len(texts) // 2)
            if n_clusters < 2:
                return {}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # 分析聚类结果
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[int(label)].append(texts[i][:50])  # 限制文本长度
            
            # 提取关键词
            feature_names = vectorizer.get_feature_names_out()
            cluster_keywords = {}
            
            for i in range(n_clusters):
                center = kmeans.cluster_centers_[i]
                top_indices = center.argsort()[-5:][::-1]  # 前5个关键词
                cluster_keywords[i] = [feature_names[idx] for idx in top_indices]
            
            return {
                'n_clusters': n_clusters,
                'cluster_keywords': cluster_keywords,
                'cluster_sizes': {str(k): len(v) for k, v in clusters.items()}
            }
            
        except Exception as e:
            logger.error(f"话题聚类分析失败: {e}")
            return {}

    async def analyze_group_sentiment_trend(self, group_id: str) -> Dict[str, Any]:
        """分析群聊情感趋势"""
        try:
            cache_key = f"sentiment_{group_id}"
            if self._check_cache(cache_key):
                return self.analysis_cache[cache_key]['data']
            
            # 获取最近消息（限制数量）
            recent_messages = await self._get_recent_group_messages(group_id, limit=self.max_sample_size)
            
            if not recent_messages:
                return {}
            
            # 简单情感分析（基于关键词）
            sentiment_trend = self._analyze_sentiment_keywords(recent_messages)
            
            # 活跃度分析
            activity_trend = self._analyze_activity_trend(recent_messages)
            
            result = {
                'sentiment_trend': sentiment_trend,
                'activity_trend': activity_trend,
                'analysis_time': datetime.now().isoformat(),
                'sample_size': len(recent_messages)
            }
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"分析群聊情感趋势失败: {e}")
            return {}

    async def _get_recent_group_messages(self, group_id: str, limit: int) -> List[Dict[str, Any]]:
        """获取群聊最近消息"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT message, timestamp, sender_id
                FROM raw_messages 
                WHERE timestamp > ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (time.time() - 3600 * 6, limit))  # 最近6小时
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'message': row,
                    'timestamp': row[1],
                    'sender_id': row
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"获取群聊最近消息失败: {e}")
            return []

    def _analyze_sentiment_keywords(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """基于关键词的简单情感分析"""
        positive_keywords = ['哈哈', '好的', '谢谢', '赞', '棒', '开心', '高兴', '😊', '👍', '❤️']
        negative_keywords = ['不行', '差', '烦', '无聊', '生气', '😢', '😡', '💔']
        
        positive_count = 0
        negative_count = 0
        total_messages = len(messages)
        
        for msg in messages:
            text = msg['message'].lower()
            for keyword in positive_keywords:
                if keyword in text:
                    positive_count += 1
                    break
            for keyword in negative_keywords:
                if keyword in text:
                    negative_count += 1
                    break
        
        return {
            'positive_ratio': positive_count / max(total_messages, 1),
            'negative_ratio': negative_count / max(total_messages, 1),
            'neutral_ratio': (total_messages - positive_count - negative_count) / max(total_messages, 1)
        }

    def _analyze_activity_trend(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析活跃度趋势"""
        if not messages:
            return {}
        
        # 按小时分组统计
        hourly_counts = defaultdict(int)
        for msg in messages:
            hour = datetime.fromtimestamp(msg['timestamp']).hour
            hourly_counts[hour] += 1
        
        # 计算趋势
        hours = sorted(hourly_counts.keys())
        counts = [hourly_counts[hour] for hour in hours]
        
        if len(counts) >= 3:
            # 简单线性趋势计算
            x = np.array(range(len(counts)))
            y = np.array(counts)
            trend_slope = np.polyfit(x, y, 1)
        else:
            trend_slope = 0
        
        return {
            'hourly_activity': dict(hourly_counts),
            'trend_slope': float(trend_slope),
            'peak_hour': max(hourly_counts.items(), key=lambda x: x) if hourly_counts else None[1],
            'total_activity': sum(counts)
        }

    def _check_cache(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.analysis_cache:
            return False
        
        cache_time = self.analysis_cache[cache_key]['timestamp']
        return time.time() - cache_time < self.cache_timeout

    def _cache_result(self, cache_key: str, data: Dict[str, Any]):
        """缓存分析结果"""
        self.analysis_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        # 清理过期缓存
        current_time = time.time()
        expired_keys = [
            key for key, value in self.analysis_cache.items()
            if current_time - value['timestamp'] > self.cache_timeout
        ]
        
        for key in expired_keys:
            del self.analysis_cache[key]

    async def get_analysis_summary(self, group_id: str) -> Dict[str, Any]:
        """获取分析摘要"""
        try:
            # 获取群统计
            group_stats = await self.db_manager.get_group_statistics(group_id)
            
            # 获取情感趋势
            sentiment_trend = await self.analyze_group_sentiment_trend(group_id)
            
            # 获取最活跃用户
            active_users = await self._get_most_active_users(group_id, limit=5)
            
            return {
                'group_statistics': group_stats,
                'sentiment_analysis': sentiment_trend,
                'active_users': active_users,
                'analysis_capabilities': {
                    'sklearn_available': SKLEARN_AVAILABLE,
                    'max_sample_size': self.max_sample_size,
                    'cache_status': len(self.analysis_cache)
                }
            }
            
        except Exception as e:
            logger.error(f"获取分析摘要失败: {e}")
            return {}

    async def _get_most_active_users(self, group_id: str, limit: int) -> List[Dict[str, Any]]:
        """获取最活跃用户"""
        try:
            conn = await self.db_manager.get_group_connection(group_id)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sender_id, sender_name, COUNT(*) as message_count
                FROM raw_messages 
                WHERE timestamp > ?
                GROUP BY sender_id
                ORDER BY message_count DESC
                LIMIT ?
            ''', (time.time() - 86400, limit))  # 最近24小时
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'user_id': row,
                    'user_name': row[1],
                    'message_count': row
                })
            
            return users
            
        except Exception as e:
            logger.error(f"获取最活跃用户失败: {e}")
            return []
