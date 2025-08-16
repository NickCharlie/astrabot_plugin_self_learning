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
    """分群数据库管理器"""
    
    def __init__(self, config: PluginConfig, context=None):
        self.config = config
        self.context = context
        self.db_connections: Dict[str, sqlite3.Connection] = {}
        self.data_dir = os.path.join(config.data_dir, "group_databases")
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("数据库管理器初始化完成")

    def get_group_db_path(self, group_id: str) -> str:
        """获取群数据库文件路径"""
        return os.path.join(self.data_dir, f"{group_id}_ID.db")

    async def get_group_connection(self, group_id: str) -> sqlite3.Connection:
        """获取群数据库连接"""
        if group_id not in self.db_connections:
            db_path = self.get_group_db_path(group_id)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            await self._init_group_database(conn)
            self.db_connections[group_id] = conn
            logger.info(f"已创建群 {group_id} 的数据库连接")
        
        return self.db_connections[group_id]

    async def _init_group_database(self, conn: sqlite3.Connection):
        """初始化群数据库表结构"""
        cursor = conn.cursor()
        
        try:
            # 原始消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    platform TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 筛选消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS filtered_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_message_id INTEGER,
                    message TEXT NOT NULL,
                    sender_id TEXT,
                    confidence REAL,
                    filter_reason TEXT,
                    timestamp REAL NOT NULL,
                    used_for_learning BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (raw_message_id) REFERENCES raw_messages (id)
                )
            ''')
            
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
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_messages_sender ON raw_messages(sender_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_messages_timestamp ON raw_messages(timestamp)')
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

    async def close_all_connections(self):
        """关闭所有数据库连接"""
        for group_id, conn in self.db_connections.items():
            try:
                conn.close()
                logger.info(f"已关闭群 {group_id} 的数据库连接")
            except Exception as e:
                logger.error(f"关闭数据库连接失败: {e}")
        
        self.db_connections.clear()
