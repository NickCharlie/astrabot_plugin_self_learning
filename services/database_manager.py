"""
数据库管理器 - 管理分群数据库和数据持久化
"""
import os
import json
import sqlite3
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict

from astrbot.api import logger

from ..config import PluginConfig
from ..exceptions import DataStorageError


class DatabaseManager:
    """数据库管理器 - 管理分群数据库和全局消息数据库的数据持久化"""
    
    def __init__(self, config: PluginConfig, context=None):
        self.config = config
        self.context = context
        self.group_db_connections: Dict[str, sqlite3.Connection] = {}
        self.group_data_dir = os.path.join(config.data_dir, "group_databases")
        self.messages_db_path = config.messages_db_path
        self.messages_db_connection: Optional[sqlite3.Connection] = None
        
        # 确保数据目录存在
        os.makedirs(self.group_data_dir, exist_ok=True)
        
        # 初始化全局消息数据库
        self._init_messages_database()
        
        logger.info("数据库管理器初始化完成")

    def _get_messages_db_connection(self) -> sqlite3.Connection:
        """获取全局消息数据库连接"""
        if self.messages_db_connection is None:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.messages_db_path), exist_ok=True)
            self.messages_db_connection = sqlite3.connect(self.messages_db_path, check_same_thread=False)
        return self.messages_db_connection

    def _init_messages_database(self):
        """初始化全局消息SQLite数据库"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        
        try:
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
                    quality_scores TEXT, -- 新增字段，存储JSON字符串
                    FOREIGN KEY (raw_message_id) REFERENCES raw_messages (id)
                )
            ''')
            
            # 检查并添加 quality_scores 列（如果不存在）
            cursor.execute("PRAGMA table_info(filtered_messages)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'quality_scores' not in columns:
                cursor.execute("ALTER TABLE filtered_messages ADD COLUMN quality_scores TEXT")
                logger.info("已为 filtered_messages 表添加 quality_scores 列。")

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
            logger.info("全局消息数据库初始化完成")
            
        except Exception as e:
            logger.error(f"全局消息数据库初始化失败: {e}")
            raise DataStorageError(f"全局消息数据库初始化失败: {str(e)}")

    def get_group_db_path(self, group_id: str) -> str:
        """获取群数据库文件路径"""
        return os.path.join(self.group_data_dir, f"{group_id}_ID.db")

    async def get_group_connection(self, group_id: str) -> sqlite3.Connection:
        """获取群数据库连接"""
        if group_id not in self.group_db_connections:
            db_path = self.get_group_db_path(group_id)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            await self._init_group_database(conn)
            self.group_db_connections[group_id] = conn
            logger.info(f"已创建群 {group_id} 的数据库连接")
        
        return self.group_db_connections[group_id]

    async def _init_group_database(self, conn: sqlite3.Connection):
        """初始化群数据库表结构"""
        cursor = conn.cursor()
        
        try:
            # 原始消息表 (群数据库中不再存储原始消息，由全局消息数据库统一管理)
            # 筛选消息表 (群数据库中不再存储筛选消息，由全局消息数据库统一管理)
            
            # 用户画像表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    qq_id TEXT PRIMARY KEY,
                    qq_name TEXT,
                    nicknames TEXT, -- JSON格式存储
                    activity_pattern TEXT, -- JSON格式存储活动模式
                    communication_style TEXT, -- JSON格式存储沟通风格
                    topic_preferences TEXT, -- JSON格式存储话题偏好
                    emotional_tendency TEXT, -- JSON格式存储情感倾向
                    last_active REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 社交关系表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS social_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_user TEXT NOT NULL,
                    to_user TEXT NOT NULL,
                    relation_type TEXT NOT NULL, -- mention, reply, frequent_interaction
                    strength REAL NOT NULL,
                    frequency INTEGER NOT NULL,
                    last_interaction REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(from_user, to_user, relation_type)
                )
            ''')
            
            # 风格档案表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS style_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_name TEXT NOT NULL,
                    vocabulary_richness REAL,
                    sentence_complexity REAL,
                    emotional_expression REAL,
                    interaction_tendency REAL,
                    topic_diversity REAL,
                    formality_level REAL,
                    creativity_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 人格备份表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persona_backups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backup_name TEXT NOT NULL,
                    original_persona TEXT, -- JSON格式存储
                    imitation_dialogues TEXT, -- JSON格式存储模仿对话
                    backup_reason TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 学习会话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    messages_processed INTEGER DEFAULT 0,
                    filtered_messages INTEGER DEFAULT 0,
                    style_updates INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0.0,
                    success BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_social_relations_from_user ON social_relations(from_user)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_social_relations_to_user ON social_relations(to_user)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_profiles_active ON user_profiles(last_active)')
            
            conn.commit()
            logger.debug("群数据库表结构初始化完成")
            
        except Exception as e:
            logger.error(f"初始化群数据库失败: {e}")
            raise DataStorageError(f"初始化群数据库失败: {str(e)}")

    async def save_user_profile(self, group_id: str, profile_data: Dict[str, Any]):
        """保存用户画像到数据库"""
        conn = await self.get_group_connection(group_id)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (qq_id, qq_name, nicknames, activity_pattern, communication_style, 
                 topic_preferences, emotional_tendency, last_active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile_data['qq_id'],
                profile_data.get('qq_name', ''),
                json.dumps(profile_data.get('nicknames', []), ensure_ascii=False),
                json.dumps(profile_data.get('activity_pattern', {}), ensure_ascii=False),
                json.dumps(profile_data.get('communication_style', {}), ensure_ascii=False),
                json.dumps(profile_data.get('topic_preferences', {}), ensure_ascii=False),
                json.dumps(profile_data.get('emotional_tendency', {}), ensure_ascii=False),
                time.time(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"保存用户画像失败: {e}")
            raise DataStorageError(f"保存用户画像失败: {str(e)}")

    async def load_user_profile(self, group_id: str, qq_id: str) -> Optional[Dict[str, Any]]:
        """从数据库加载用户画像"""
        conn = await self.get_group_connection(group_id)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT qq_id, qq_name, nicknames, activity_pattern, communication_style,
                       topic_preferences, emotional_tendency, last_active
                FROM user_profiles WHERE qq_id = ?
            ''', (qq_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'qq_id': row,
                'qq_name': row[1],
                'nicknames': json.loads(row) if row else [],
                'activity_pattern': json.loads(row) if row else {},
                'communication_style': json.loads(row) if row else {},
                'topic_preferences': json.loads(row) if row else {},
                'emotional_tendency': json.loads(row) if row else {},
                'last_active': row
            }
            
        except Exception as e:
            logger.error(f"加载用户画像失败: {e}")
            return None

    async def save_social_relation(self, group_id: str, relation_data: Dict[str, Any]):
        """保存社交关系到数据库"""
        conn = await self.get_group_connection(group_id)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO social_relations 
                (from_user, to_user, relation_type, strength, frequency, last_interaction, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                relation_data['from_user'],
                relation_data['to_user'],
                relation_data['relation_type'],
                relation_data['strength'],
                relation_data['frequency'],
                relation_data['last_interaction'],
                datetime.now().isoformat()
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"保存社交关系失败: {e}")
            raise DataStorageError(f"保存社交关系失败: {str(e)}")

    async def load_social_graph(self, group_id: str) -> List[Dict[str, Any]]:
        """加载完整社交图谱"""
        conn = await self.get_group_connection(group_id)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT from_user, to_user, relation_type, strength, frequency, last_interaction
                FROM social_relations ORDER BY strength DESC
            ''')
            
            relations = []
            for row in cursor.fetchall():
                relations.append({
                    'from_user': row,
                    'to_user': row[1],
                    'relation_type': row,
                    'strength': row,
                    'frequency': row,
                    'last_interaction': row
                })
            
            return relations
            
        except Exception as e:
            logger.error(f"加载社交图谱失败: {e}")
            return []

    async def backup_persona(self, group_id: str, backup_data: Dict[str, Any]) -> int:
        """备份人格数据"""
        conn = await self.get_group_connection(group_id)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO persona_backups (backup_name, original_persona, imitation_dialogues, backup_reason)
                VALUES (?, ?, ?, ?)
            ''', (
                backup_data['backup_name'],
                json.dumps(backup_data['original_persona'], ensure_ascii=False),
                json.dumps(backup_data.get('imitation_dialogues', []), ensure_ascii=False),
                backup_data.get('backup_reason', 'Auto backup before update')
            ))
            
            backup_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"人格数据已备份，备份ID: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"备份人格数据失败: {e}")
            raise DataStorageError(f"备份人格数据失败: {str(e)}")

    async def get_persona_backups(self, group_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取人格备份列表"""
        conn = await self.get_group_connection(group_id)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, backup_name, backup_reason, created_at
                FROM persona_backups 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            backups = []
            for row in cursor.fetchall():
                backups.append({
                    'id': row,
                    'backup_name': row[1],
                    'backup_reason': row,
                    'created_at': row
                })
            
            return backups
            
        except Exception as e:
            logger.error(f"获取人格备份列表失败: {e}")
            return []

    async def restore_persona_backup(self, group_id: str, backup_id: int) -> Optional[Dict[str, Any]]:
        """恢复人格备份"""
        conn = await self.get_group_connection(group_id)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT original_persona, imitation_dialogues 
                FROM persona_backups 
                WHERE id = ?
            ''', (backup_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'original_persona': json.loads(row),
                'imitation_dialogues': json.loads(row)[1]
            }
            
        except Exception as e:
            logger.error(f"恢复人格备份失败: {e}")
            return None

    async def get_group_statistics(self, group_id: str) -> Dict[str, Any]:
        """获取群统计信息"""
        conn = await self.get_group_connection(group_id)
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # 消息统计
            cursor.execute('SELECT COUNT(*) FROM raw_messages')
            stats['total_messages'] = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM filtered_messages')
            stats['filtered_messages'] = cursor.fetchone()
            
            # 用户统计
            cursor.execute('SELECT COUNT(*) FROM user_profiles')
            stats['total_users'] = cursor.fetchone()
            
            # 社交关系统计
            cursor.execute('SELECT COUNT(*) FROM social_relations')
            stats['total_relations'] = cursor.fetchone()
            
            # 备份统计
            cursor.execute('SELECT COUNT(*) FROM persona_backups')
            stats['total_backups'] = cursor.fetchone()
            
            return stats
            
        except Exception as e:
            logger.error(f"获取群统计失败: {e}")
            return {}

    async def collect_message(self, message_data: Dict[str, Any]) -> bool:
        """收集消息到缓存并刷新到数据库"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            # 验证消息数据
            required_fields = ['sender_id', 'message', 'timestamp']
            for field in required_fields:
                if field not in message_data:
                    logger.warning(f"消息数据缺少必要字段: {field}")
                    return False
            
            insert_data = (
                message_data['sender_id'],
                message_data.get('sender_name', ''),
                message_data['message'],
                message_data.get('group_id', ''),
                message_data.get('platform', ''),
                message_data['timestamp']
            )
            
            cursor.execute('''
                INSERT INTO raw_messages 
                (sender_id, sender_name, message, group_id, platform, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', insert_data)
            
            conn.commit()
            logger.debug(f"已收集并插入一条消息到数据库: {message_data['sender_id']}")
            return True
            
        except Exception as e:
            logger.error(f"消息收集失败: {e}")
            raise DataStorageError(f"消息收集失败: {str(e)}")

    async def get_unprocessed_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取未处理的消息"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
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
                    'id': row[0],
                    'sender_id': row[1],
                    'sender_name': row[2],
                    'message': row[3],
                    'group_id': row[4],
                    'platform': row[5],
                    'timestamp': row[6]
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"获取未处理消息失败: {e}")
            raise DataStorageError(f"获取未处理消息失败: {str(e)}")

    async def add_filtered_message(self, filtered_data: Dict[str, Any]) -> bool:
        """添加筛选后的消息"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            quality_scores_json = json.dumps(filtered_data.get('quality_scores', {}), ensure_ascii=False)
            
            cursor.execute('''
                INSERT INTO filtered_messages 
                (raw_message_id, message, sender_id, confidence, filter_reason, timestamp, quality_scores)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                filtered_data.get('raw_message_id'),
                filtered_data['message'],
                filtered_data.get('sender_id', ''),
                filtered_data.get('confidence', 0.0),
                filtered_data.get('filter_reason', ''),
                filtered_data.get('timestamp', time.time()),
                quality_scores_json
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"添加筛选消息失败: {e}")
            raise DataStorageError(f"添加筛选消息失败: {str(e)}")

    async def mark_messages_processed(self, message_ids: List[int]):
        """标记消息为已处理"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            if not message_ids:
                return
                
            placeholders = ','.join(['?' for _ in message_ids])
            cursor.execute(f'''
                UPDATE raw_messages 
                SET processed = TRUE 
                WHERE id IN ({placeholders})
            ''', message_ids)
            
            conn.commit()
            logger.debug(f"已标记 {len(message_ids)} 条消息为已处理")
            
        except Exception as e:
            logger.error(f"标记消息处理状态失败: {e}")
            raise DataStorageError(f"标记消息处理状态失败: {str(e)}")

    async def get_filtered_messages_for_learning(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取用于学习的筛选消息"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            query = '''
                SELECT id, message, sender_id, confidence, timestamp, quality_scores
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
                quality_scores = json.loads(row[5]) if row[5] else {}
                messages.append({
                    'id': row[0],
                    'message': row[1],
                    'sender_id': row[2],
                    'confidence': row[3],
                    'timestamp': row[4],
                    'quality_scores': quality_scores
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"获取学习消息失败: {e}")
            raise DataStorageError(f"获取学习消息失败: {str(e)}")

    async def get_recent_filtered_messages(self, group_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取指定群组最近的、包含多维度评分的筛选消息。
        """
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            query = '''
                SELECT fm.id, fm.message, fm.sender_id, fm.confidence, fm.timestamp, fm.quality_scores, rm.group_id
                FROM filtered_messages fm
                JOIN raw_messages rm ON fm.raw_message_id = rm.id
                WHERE rm.group_id = ?
                ORDER BY fm.timestamp DESC
                LIMIT ?
            '''
            cursor.execute(query, (group_id, limit))
            rows = cursor.fetchall()

            messages = []
            for row in rows:
                quality_scores = json.loads(row[5]) if row[5] else {}
                messages.append({
                    'id': row[0],
                    'message': row[1],
                    'sender_id': row[2],
                    'confidence': row[3],
                    'timestamp': row[4],
                    'quality_scores': quality_scores,
                    'group_id': row[6]
                })
            
            return messages

        except Exception as e:
            logger.error(f"获取最近筛选消息失败: {e}")
            return []

    async def get_messages_statistics(self) -> Dict[str, Any]:
        """获取消息收集统计信息"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            # 统计原始消息
            cursor.execute('SELECT COUNT(*) FROM raw_messages')
            total_raw = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM raw_messages WHERE processed = FALSE')
            unprocessed = cursor.fetchone()[0]
            
            # 统计筛选消息
            cursor.execute('SELECT COUNT(*) FROM filtered_messages')
            total_filtered = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM filtered_messages WHERE used_for_learning = FALSE')
            unused_filtered = cursor.fetchone()[0]
            
            # 统计学习批次
            cursor.execute('SELECT COUNT(*) FROM learning_batches')
            total_batches = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM learning_batches WHERE success = TRUE')
            successful_batches = cursor.fetchone()[0]
            
            return {
                'raw_messages': total_raw,
                'unprocessed_messages': unprocessed,
                'filtered_messages': total_filtered,
                'unused_filtered_messages': unused_filtered,
                'learning_batches': total_batches,
                'successful_batches': successful_batches,
            }
            
        except Exception as e:
            logger.error(f"获取消息统计信息失败: {e}")
            return {}

    async def export_messages_learning_data(self) -> Dict[str, Any]:
        """导出消息学习数据"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            # 导出筛选后的消息
            cursor.execute('''
                SELECT message, sender_id, confidence, timestamp, used_for_learning, quality_scores
                FROM filtered_messages 
                ORDER BY timestamp DESC
            ''')
            
            filtered_messages = []
            for row in cursor.fetchall():
                quality_scores = json.loads(row[5]) if row[5] else {}
                filtered_messages.append({
                    'message': row[0],
                    'sender_id': row[1],
                    'confidence': row[2],
                    'timestamp': row[3],
                    'used_for_learning': bool(row[4]),
                    'quality_scores': quality_scores
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
                    'batch_name': row[0],
                    'start_time': row[1],
                    'end_time': row[2],
                    'message_count': row[3],
                    'filtered_count': row[4],
                    'success': bool(row[5]),
                    'error_message': row[6]
                })
            
            return {
                'export_time': datetime.now().isoformat(),
                'filtered_messages': filtered_messages,
                'learning_batches': learning_batches,
                'statistics': await self.get_messages_statistics()
            }
            
        except Exception as e:
            logger.error(f"导出消息学习数据失败: {e}")
            raise DataStorageError(f"导出消息学习数据失败: {str(e)}")

    async def clear_all_messages_data(self):
        """清空所有消息数据"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM learning_batches')
            cursor.execute('DELETE FROM filtered_messages')
            cursor.execute('DELETE FROM raw_messages')
            
            conn.commit()
            logger.info("所有消息数据已清空")
            
        except Exception as e:
            logger.error(f"清空消息数据失败: {e}")
            raise DataStorageError(f"清空消息数据失败: {str(e)}")

    async def create_learning_batch(self, batch_name: str) -> int:
        """创建学习批次记录"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO learning_batches (batch_name, start_time)
                VALUES (?, ?)
            ''', (batch_name, datetime.now().isoformat()))
            
            batch_id = cursor.lastrowid
            conn.commit()
            return batch_id
            
        except Exception as e:
            logger.error(f"创建学习批次失败: {e}")
            raise DataStorageError(f"创建学习批次失败: {str(e)}")

    async def update_learning_batch(self, batch_id: int, **kwargs):
        """更新学习批次信息"""
        conn = self._get_messages_db_connection()
        cursor = conn.cursor()
        try:
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
            
        except Exception as e:
            logger.error(f"更新学习批次失败: {e}")
            raise DataStorageError(f"更新学习批次失败: {str(e)}")

    async def close_all_connections(self):
        """关闭所有数据库连接"""
        for group_id, conn in self.group_db_connections.items():
            try:
                conn.close()
                logger.info(f"已关闭群 {group_id} 的数据库连接")
            except Exception as e:
                logger.error(f"关闭群数据库连接失败: {e}")
        self.group_db_connections.clear()

        if self.messages_db_connection:
            try:
                self.messages_db_connection.close()
                logger.info("已关闭全局消息数据库连接")
            except Exception as e:
                logger.error(f"关闭全局消息数据库连接失败: {e}")
            self.messages_db_connection = None
