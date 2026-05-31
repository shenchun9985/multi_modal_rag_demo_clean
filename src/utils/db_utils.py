import sqlite3
import json
from datetime import datetime
from src.utils.logger import get_logger
from config import DB_PATH

logger = get_logger(__name__)

class SessionDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库：创建 sessions 表和 messages 表"""
        with sqlite3.connect(self.db_path) as conn:
            # 会话元数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 消息表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            ''')
            # 索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)')
            logger.info("数据库初始化完成（sessions + messages 表）")

    # ========== 会话管理 ==========
    def create_session(self, session_id: str = None, title: str = "") -> str:
        """创建新会话，返回 session_id（如果未提供则自动生成）"""
        if not session_id:
            session_id = f"session_{int(datetime.now().timestamp())}"
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO sessions (session_id, title) VALUES (?, ?)",
                    (session_id, title or session_id)
                )
            except sqlite3.IntegrityError:
                # 已存在则更新 updated_at
                conn.execute(
                    "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (session_id,)
                )
        return session_id

    def delete_session(self, session_id: str):
        """删除会话及其所有消息（CASCADE 自动删除消息）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def get_all_sessions(self, limit: int = 100):
        """获取所有会话，按最后更新时间倒序"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [{"session_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

    def update_session_title(self, session_id: str, title: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id))

    def touch_session(self, session_id: str):
        """更新会话的最后活跃时间"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))

    # ========== 消息管理 ==========
    def save_message(self, session_id: str, role: str, content):
        """保存消息，并确保会话存在（自动创建）"""
        # 先确保会话存在
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, title) VALUES (?, ?)",
                (session_id, session_id)
            )
            # 更新 updated_at
            conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))
        # 保存消息
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )

    def load_history(self, session_id: str, max_turns: int = 20):
        """加载最近的对话历史（user+assistant 成对）"""
        with sqlite3.connect(self.db_path) as conn:
            # 取最近 2*max_turns 条消息
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, 2 * max_turns)
            ).fetchall()
        rows.reverse()
        history = []
        for role, content in rows:
            try:
                content = json.loads(content)
            except:
                pass
            history.append({"role": role, "content": content})
        return history

    def clear_session(self, session_id: str):
        """清空会话（删除所有消息，但保留会话元数据）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))