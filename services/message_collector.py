"""
消息收集服务 - 负责收集、存储和管理用户消息数据
"""
import os
import json
import sqlite3
import asyncio
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict

from astrbot.api import logger
from astrbot.api.star import Context

from ..config import PluginConfig
from ..exceptions import MessageCollectionError, DataStorageError


class MessageCollectorService:
    """消息收集服务类"""
    
    def __init__(self, config: PluginConfig, context: Context):
        self.config = config
        self.context = context
        self.db_path = config.messages_db_path
        
        # 初始化数据库
        self._init_database()
        
        # 消息缓存（用于批量写入优化）
        self._message_cache = []
        self._cache_size_limit = 100
        self._last_flush_time = time.time()
        self._flush_interval = 30  # 30秒强制刷新一次
        
        logger.info(f"消息收集服务初始化完成，数据库路径: {self.db_path}")

    def _init_database(self):
        """初始化SQLite数据库"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建原始消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT,
                    message TEXT NOT NULL,
                    group_id TEXT,
                    platform TEXT,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # 创建筛选后消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS filtered_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_message_id INTEGER,
                    message TEXT NOT NULL,
                    sender_id TEXT,
                    confidence REAL,
                    filter_reason TEXT,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    used_for_learning BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (raw_message_id) REFERENCES raw_messages (id)
                )
            ''')
            
            # 创建学习批次表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_name TEXT UNIQUE,
                    start_time DATETIME,
                    end_time DATETIME,
                    message_count INTEGER,
                    filtered_count INTEGER,
                    success BOOLEAN,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_messages_timestamp ON raw_messages(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_messages_sender ON raw_messages(sender_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_messages_processed ON raw_messages(processed)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_filtered_messages_confidence ON filtered_messages(confidence)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_filtered_messages_used ON filtered_messages(used_for_learning)')
            
            conn.commit()
            conn.close()
            
            logger.info("数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DataStorageError(f"数据库初始化失败: {str(e)}")

    async def collect_message(self, message_data: Dict[str, Any]) -> bool:
        """收集消息到缓存"""
        try:
            # 验证消息数据
            required_fields = ['sender_id', 'message', 'timestamp']
            for field in required_fields:
                if field not in message_data:
                    logger.warning(f"消息数据缺少必要字段: {field}")
                    return False
            
            # 添加到缓存
            self._message_cache.append(message_data)
            
            # 检查是否需要刷新缓存
            if (len(self._message_cache) >= self._cache_size_limit or 
                time.time() - self._last_flush_time > self._flush_interval):
                await self._flush_message_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"消息收集失败: {e}")
            raise MessageCollectionError(f"消息收集失败: {str(e)}")

    async def _flush_message_cache(self):
        """刷新消息缓存到数据库"""
        if not self._message_cache:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 批量插入消息
            insert_data = []
            for msg in self._message_cache:
                insert_data.append((
                    msg['sender_id'],
                    msg.get('sender_name', ''),
                    msg['message'],
                    msg.get('group_id', ''),
                    msg.get('platform', ''),
                    msg['timestamp']
                ))
            
            cursor.executemany('''
                INSERT INTO raw_messages 
                (sender_id, sender_name, message, group_id, platform, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', insert_data)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"已刷新 {len(self._message_cache)} 条消息到数据库")
            
            # 清空缓存
            self._message_cache.clear()
            self._last_flush_time = time.time()
            
        except Exception as e:
            logger.error(f"消息缓存刷新失败: {e}")
            raise DataStorageError(f"消息缓存刷新失败: {str(e)}")

    async def get_unprocessed_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取未处理的消息"""
        try:
            # 先刷新缓存
            await self._flush_message_cache()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, sender_id, sender_name, message, group_id, platform, timestamp
                FROM raw_messages 
                WHERE processed = FALSE 
                ORDER BY timestamp ASC
            '''
            
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                messages.append({
                    'id': row,
                    'sender_id': row[1],
                    'sender_name': row,
                    'message': row,
                    'group_id': row,
                    'platform': row,
                    'timestamp': row
                })
            
            conn.close()
            return messages
            
        except Exception as e:
            logger.error(f"获取未处理消息失败: {e}")
            raise DataStorageError(f"获取未处理消息失败: {str(e)}")

    async def add_filtered_message(self, filtered_data: Dict[str, Any]) -> bool:
        """添加筛选后的消息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO filtered_messages 
                (raw_message_id, message, sender_id, confidence, filter_reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                filtered_data.get('raw_message_id'),
                filtered_data['message'],
                filtered_data.get('sender_id', ''),
                filtered_data.get('confidence', 0.0),
                filtered_data.get('filter_reason', ''),
                filtered_data.get('timestamp', time.time())
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"添加筛选消息失败: {e}")
            raise DataStorageError(f"添加筛选消息失败: {str(e)}")

    async def mark_messages_processed(self, message_ids: List[int]):
        """标记消息为已处理"""
        try:
            if not message_ids:
                return
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in message_ids])
            cursor.execute(f'''
                UPDATE raw_messages 
                SET processed = TRUE 
                WHERE id IN ({placeholders})
            ''', message_ids)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"已标记 {len(message_ids)} 条消息为已处理")
            
        except Exception as e:
            logger.error(f"标记消息处理状态失败: {e}")
            raise DataStorageError(f"标记消息处理状态失败: {str(e)}")

    async def get_filtered_messages_for_learning(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取用于学习的筛选消息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, message, sender_id, confidence, timestamp
                FROM filtered_messages 
                WHERE used_for_learning = FALSE 
                ORDER BY confidence DESC, timestamp DESC
            '''
            
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                messages.append({
                    'id': row,
                    'message': row[1],
                    'sender_id': row,
                    'confidence': row,
                    'timestamp': row
                })
            
            conn.close()
            return messages
            
        except Exception as e:
            logger.error(f"获取学习消息失败: {e}")
            raise DataStorageError(f"获取学习消息失败: {str(e)}")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取收集统计信息"""
        try:
            # 先刷新缓存
            await self._flush_message_cache()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 统计原始消息
            cursor.execute('SELECT COUNT(*) FROM raw_messages')
            total_raw = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM raw_messages WHERE processed = FALSE')
            unprocessed = cursor.fetchone()
            
            # 统计筛选消息
            cursor.execute('SELECT COUNT(*) FROM filtered_messages')
            total_filtered = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM filtered_messages WHERE used_for_learning = FALSE')
            unused_filtered = cursor.fetchone()
            
            # 统计学习批次
            cursor.execute('SELECT COUNT(*) FROM learning_batches')
            total_batches = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM learning_batches WHERE success = TRUE')
            successful_batches = cursor.fetchone()
            
            conn.close()
            
            return {
                'raw_messages': total_raw,
                'unprocessed_messages': unprocessed,
                'filtered_messages': total_filtered,
                'unused_filtered_messages': unused_filtered,
                'learning_batches': total_batches,
                'successful_batches': successful_batches,
                'cache_size': len(self._message_cache)
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    async def export_learning_data(self) -> Dict[str, Any]:
        """导出学习数据"""
        try:
            await self._flush_message_cache()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 导出筛选后的消息
            cursor.execute('''
                SELECT message, sender_id, confidence, timestamp, used_for_learning
                FROM filtered_messages 
                ORDER BY timestamp DESC
            ''')
            
            filtered_messages = []
            for row in cursor.fetchall():
                filtered_messages.append({
                    'message': row,
                    'sender_id': row[1],
                    'confidence': row,
                    'timestamp': row,
                    'used_for_learning': bool(row)
                })
            
            # 导出学习批次
            cursor.execute('''
                SELECT batch_name, start_time, end_time, message_count, 
                       filtered_count, success, error_message
                FROM learning_batches 
                ORDER BY start_time DESC
            ''')
            
            learning_batches = []
            for row in cursor.fetchall():
                learning_batches.append({
                    'batch_name': row,
                    'start_time': row[1],
                    'end_time': row,
                    'message_count': row,
                    'filtered_count': row,
                    'success': bool(row),
                    'error_message': row
                })
            
            conn.close()
            
            return {
                'export_time': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'filtered_messages': filtered_messages,
                'learning_batches': learning_batches,
                'statistics': await self.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"导出学习数据失败: {e}")
            raise DataStorageError(f"导出学习数据失败: {str(e)}")

    async def clear_all_data(self):
        """清空所有数据"""
        try:
            await self._flush_message_cache()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM learning_batches')
            cursor.execute('DELETE FROM filtered_messages')
            cursor.execute('DELETE FROM raw_messages')
            
            conn.commit()
            conn.close()
            
            self._message_cache.clear()
            
            logger.info("所有学习数据已清空")
            
        except Exception as e:
            logger.error(f"清空数据失败: {e}")
            raise DataStorageError(f"清空数据失败: {str(e)}")

    async def save_state(self):
        """保存当前状态"""
        try:
            await self._flush_message_cache()
            logger.info("消息收集服务状态已保存")
            
        except Exception as e:
            logger.error(f"保存状态失败: {e}")

    async def create_learning_batch(self, batch_name: str) -> int:
        """创建学习批次记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learning_batches (batch_name, start_time)
                VALUES (?, ?)
            ''', (batch_name, datetime.now().isoformat()))
            
            batch_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return batch_id
            
        except Exception as e:
            logger.error(f"创建学习批次失败: {e}")
            raise DataStorageError(f"创建学习批次失败: {str(e)}")

    async def update_learning_batch(self, batch_id: int, **kwargs):
        """更新学习批次信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = []
            values = []
            
            if 'end_time' in kwargs:
                update_fields.append('end_time = ?')
                values.append(kwargs['end_time'])
            
            if 'message_count' in kwargs:
                update_fields.append('message_count = ?')
                values.append(kwargs['message_count'])
                
            if 'filtered_count' in kwargs:
                update_fields.append('filtered_count = ?')
                values.append(kwargs['filtered_count'])
                
            if 'success' in kwargs:
                update_fields.append('success = ?')
                values.append(kwargs['success'])
                
            if 'error_message' in kwargs:
                update_fields.append('error_message = ?')
                values.append(kwargs['error_message'])
            
            if update_fields:
                values.append(batch_id)
                query = f"UPDATE learning_batches SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, values)
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            logger.error(f"更新学习批次失败: {e}")
            raise DataStorageError(f"更新学习批次失败: {str(e)}")
