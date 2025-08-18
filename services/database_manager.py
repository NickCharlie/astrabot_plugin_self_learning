"""
数据库管理器 - 管理分群数据库和数据持久化
"""
import os
import json
import aiosqlite
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict

from astrbot.api import logger

from ..config import PluginConfig
from ..exceptions import DataStorageError
from ..core.interfaces import IAsyncService
from ..core.patterns import AsyncServiceBase


class DatabaseManager(AsyncServiceBase):
    """数据库管理器 - 管理分群数据库和全局消息数据库的数据持久化"""
    
    def __init__(self, config: PluginConfig, context=None):
        super().__init__("database_manager") # 调用基类构造函数
        self.config = config
        self.context = context
        self.group_db_connections: Dict[str, aiosqlite.Connection] = {}
        self.group_data_dir = os.path.join(config.data_dir, "group_databases")
        self.messages_db_path = config.messages_db_path
        self.messages_db_connection: Optional[aiosqlite.Connection] = None
        
        # 确保数据目录存在
        os.makedirs(self.group_data_dir, exist_ok=True)
        
        self._logger.info("数据库管理器初始化完成") # 使用基类的logger

    async def _do_start(self) -> bool:
        """启动服务时初始化数据库"""
        try:
            await self._init_messages_database()
            self._logger.info("全局消息数据库初始化成功")
            return True
        except Exception as e:
            self._logger.error(f"全局消息数据库初始化失败: {e}", exc_info=True)
            return False

    async def _do_stop(self) -> bool:
        """停止服务时关闭所有数据库连接"""
        try:
            await self.close_all_connections()
            self._logger.info("所有数据库连接已关闭")
            return True
        except Exception as e:
            self._logger.error(f"关闭数据库连接失败: {e}", exc_info=True)
            return False

    async def _get_messages_db_connection(self) -> aiosqlite.Connection:
        """获取全局消息数据库连接"""
        if self.messages_db_connection is None:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.messages_db_path), exist_ok=True)
            self.messages_db_connection = await aiosqlite.connect(self.messages_db_path)
        return self.messages_db_connection

    async def _init_messages_database(self):
        """初始化全局消息SQLite数据库"""
        conn = await self._get_messages_db_connection()
        cursor = await conn.cursor()
        
        try:
            # 创建原始消息表
            await cursor.execute('''
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
            await cursor.execute('''
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
            await cursor.execute("PRAGMA table_info(filtered_messages)")
            columns = [col[1] for col in await cursor.fetchall()]
            if 'quality_scores' not in columns:
                await cursor.execute("ALTER TABLE filtered_messages ADD COLUMN quality_scores TEXT")
                logger.info("已为 filtered_messages 表添加 quality_scores 列。")

            # 创建学习批次表
            await cursor.execute('''
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
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_messages_timestamp ON raw_messages(timestamp)')
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_messages_sender ON raw_messages(sender_id)')
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_messages_processed ON raw_messages(processed)')
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_filtered_messages_confidence ON filtered_messages(confidence)')
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_filtered_messages_used ON filtered_messages(used_for_learning)')
            
            await conn.commit()
            logger.info("全局消息数据库初始化完成")
            
        except aiosqlite.Error as e:
            logger.error(f"全局消息数据库初始化失败: {e}", exc_info=True)
            raise DataStorageError(f"全局消息数据库初始化失败: {str(e)}")

    def get_group_db_path(self, group_id: str) -> str:
        """获取群数据库文件路径"""
        return os.path.join(self.group_data_dir, f"{group_id}_ID.db")

    async def get_group_connection(self, group_id: str) -> aiosqlite.Connection:
        """获取群数据库连接"""
        if group_id not in self.group_db_connections:
            db_path = self.get_group_db_path(group_id)
            conn = await aiosqlite.connect(db_path)
            await self._init_group_database(conn)
            self.group_db_connections[group_id] = conn
            logger.info(f"已创建群 {group_id} 的数据库连接")
        
        return self.group_db_connections[group_id]

    async def _init_group_database(self, conn: aiosqlite.Connection):
        """初始化群数据库表结构"""
        cursor = await conn.cursor()
        
        try:
            # 原始消息表 (群数据库中不再存储原始消息，由全局消息数据库统一管理)
            # 筛选消息表 (群数据库中不再存储筛选消息，由全局消息数据库统一管理)
            
            # 用户画像表
            await cursor.execute(''' 
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
            await cursor.execute(''' 
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
            await cursor.execute(''' 
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
            await cursor.execute(''' 
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
            await cursor.execute(''' 
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
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_social_relations_from_user ON social_relations(from_user)') 
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_social_relations_to_user ON social_relations(to_user)') 
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_profiles_active ON user_profiles(last_active)') 
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_style_profiles_name ON style_profiles(profile_name)')
            
            await conn.commit() 
            logger.debug("群数据库表结构初始化完成") 
            
        except aiosqlite.Error as e: 
            logger.error(f"初始化群数据库失败: {e}", exc_info=True) 
            raise DataStorageError(f"初始化群数据库失败: {str(e)}")

    async def save_style_profile(self, group_id: str, profile_data: Dict[str, Any]):
        """保存风格档案到数据库"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()

        try:
            await cursor.execute('''
                INSERT OR REPLACE INTO style_profiles
                (profile_name, vocabulary_richness, sentence_complexity, emotional_expression,
                 interaction_tendency, topic_diversity, formality_level, creativity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile_data['profile_name'],
                profile_data.get('vocabulary_richness'),
                profile_data.get('sentence_complexity'),
                profile_data.get('emotional_expression'),
                profile_data.get('interaction_tendency'),
                profile_data.get('topic_diversity'),
                profile_data.get('formality_level'),
                profile_data.get('creativity_score')
            ))
            await conn.commit()
            logger.debug(f"风格档案 '{profile_data['profile_name']}' 已保存到群 {group_id} 数据库。")
        except aiosqlite.Error as e:
            logger.error(f"保存风格档案失败: {e}", exc_info=True)
            raise DataStorageError(f"保存风格档案失败: {str(e)}")

    async def load_style_profile(self, group_id: str, profile_name: str) -> Optional[Dict[str, Any]]:
        """从数据库加载风格档案"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()

        try:
            await cursor.execute('''
                SELECT profile_name, vocabulary_richness, sentence_complexity, emotional_expression,
                       interaction_tendency, topic_diversity, formality_level, creativity_score
                FROM style_profiles WHERE profile_name = ?
            ''', (profile_name,))
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                'profile_name': row[0],
                'vocabulary_richness': row[1],
                'sentence_complexity': row[2],
                'emotional_expression': row[3],
                'interaction_tendency': row[4],
                'topic_diversity': row[5],
                'formality_level': row[6],
                'creativity_score': row[7]
            }
        except aiosqlite.Error as e:
            logger.error(f"加载风格档案失败: {e}", exc_info=True)
            return None

    async def save_user_profile(self, group_id: str, profile_data: Dict[str, Any]):
        """保存用户画像到数据库"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()
        
        try:
            await cursor.execute('''
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
            
            await conn.commit()
            
        except aiosqlite.Error as e:
            logger.error(f"保存用户画像失败: {e}", exc_info=True)
            raise DataStorageError(f"保存用户画像失败: {str(e)}")

    async def load_user_profile(self, group_id: str, qq_id: str) -> Optional[Dict[str, Any]]:
        """从数据库加载用户画像"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()
        
        try:
            await cursor.execute('''
                SELECT qq_id, qq_name, nicknames, activity_pattern, communication_style,
                       topic_preferences, emotional_tendency, last_active
                FROM user_profiles WHERE qq_id = ?
            ''', (qq_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            return {
                'qq_id': row[0],
                'qq_name': row[1],
                'nicknames': json.loads(row[2]) if row[2] else [],
                'activity_pattern': json.loads(row[3]) if row[3] else {},
                'communication_style': json.loads(row[4]) if row[4] else {},
                'topic_preferences': json.loads(row[5]) if row[5] else {},
                'emotional_tendency': json.loads(row[6]) if row[6] else {},
                'last_active': row[7]
            }
            
        except aiosqlite.Error as e:
            logger.error(f"加载用户画像失败: {e}", exc_info=True)
            return None

    async def save_social_relation(self, group_id: str, relation_data: Dict[str, Any]):
        """保存社交关系到数据库"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()
        
        try:
            await cursor.execute(''' 
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
            
            await conn.commit()
            
        except aiosqlite.Error as e:
            logger.error(f"保存社交关系失败: {e}", exc_info=True)
            raise DataStorageError(f"保存社交关系失败: {str(e)}")

    async def load_social_graph(self, group_id: str) -> List[Dict[str, Any]]:
        """加载完整社交图谱"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()
        
        try:
            await cursor.execute('''
                SELECT from_user, to_user, relation_type, strength, frequency, last_interaction
                FROM social_relations ORDER BY strength DESC
            ''')
            
            relations = []
            for row in await cursor.fetchall():
                relations.append({
                    'from_user': row[0],
                    'to_user': row[1],
                    'relation_type': row[2],
                    'strength': row[3],
                    'frequency': row[4],
                    'last_interaction': row[5]
                })
            
            return relations
            
        except aiosqlite.Error as e:
            self._logger.error(f"加载社交图谱失败: {e}", exc_info=True)
            return []

    async def get_messages_for_replay(self, group_id: str, days: int, limit: int) -> List[Dict[str, Any]]:
        """
        从全局消息数据库获取指定群组在过去一段时间内的原始消息，用于记忆重放。
        """
        conn = await self._get_messages_db_connection()
        cursor = await conn.cursor()
        
        try:
            start_timestamp = time.time() - (days * 86400) # 转换为秒
            
            await cursor.execute('''
                SELECT id, sender_id, sender_name, message, group_id, platform, timestamp
                FROM raw_messages 
                WHERE group_id = ? AND timestamp > ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (group_id, start_timestamp, limit))
            
            messages = []
            for row in await cursor.fetchall():
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
            
        except aiosqlite.Error as e:
            self._logger.error(f"获取记忆重放消息失败: {e}", exc_info=True)
            return []

    async def backup_persona(self, group_id: str, backup_data: Dict[str, Any]) -> int:
        """备份人格数据"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()
        
        try:
            await cursor.execute(''' 
                INSERT INTO persona_backups (backup_name, original_persona, imitation_dialogues, backup_reason)
                VALUES (?, ?, ?, ?)
            ''', (
                backup_data['backup_name'],
                json.dumps(backup_data['original_persona'], ensure_ascii=False),
                json.dumps(backup_data.get('imitation_dialogues', []), ensure_ascii=False),
                backup_data.get('backup_reason', 'Auto backup before update')
            ))
            
            backup_id = cursor.lastrowid
            await conn.commit()
            
            logger.info(f"人格数据已备份，备份ID: {backup_id}")
            return backup_id
            
        except aiosqlite.Error as e:
            logger.error(f"备份人格数据失败: {e}", exc_info=True)
            raise DataStorageError(f"备份人格数据失败: {str(e)}")

    async def get_persona_backups(self, group_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的人格备份"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()
        
        try:
            await cursor.execute('''
                SELECT id, backup_name, created_at FROM persona_backups 
                ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
            
            backups = []
            for row in await cursor.fetchall():
                backups.append({
                    'id': row[0],
                    'backup_name': row[1],
                    'created_at': row[2]
                })
            
            return backups
            
        except aiosqlite.Error as e:
            logger.error(f"获取人格备份失败: {e}", exc_info=True)
            return []

    async def restore_persona(self, group_id: str, backup_id: int) -> Optional[Dict[str, Any]]:
        """从备份恢复人格数据"""
        conn = await self.get_group_connection(group_id)
        cursor = await conn.cursor()
        
        try:
            await cursor.execute('''
                SELECT backup_name, original_persona, imitation_dialogues, backup_reason 
                FROM persona_backups WHERE id = ?
            ''', (backup_id,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            return {
                'backup_name': row[0],
                'original_persona': json.loads(row[1]),
                'imitation_dialogues': json.loads(row[2]),
                'backup_reason': row[3]
            }
            
        except aiosqlite.Error as e:
            logger.error(f"恢复人格数据失败: {e}", exc_info=True)
            return None
