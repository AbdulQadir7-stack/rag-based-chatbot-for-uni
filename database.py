# database.py - SQLite database layer

import sqlite3
import bcrypt
from datetime import datetime
from collections import Counter, defaultdict

DB_NAME = "database.db"


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def add_column_if_missing(cursor, table: str, column: str, definition: str):
    cursor.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    with get_db() as conn:
        c = conn.cursor()

        # Users table
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL,
                role     TEXT    NOT NULL DEFAULT 'student'
            )
        """)

        # Query logs table
        c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER,
                username   TEXT,
                query      TEXT,
                response   TEXT,
                route      TEXT,
                allowed    INTEGER,
                timestamp  TEXT
            )
        """)

        add_column_if_missing(c, "logs", "query_id", "TEXT")
        add_column_if_missing(c, "logs", "collection", "TEXT")
        add_column_if_missing(c, "logs", "chunks_found", "INTEGER DEFAULT 0")
        add_column_if_missing(c, "logs", "top_rerank_score", "REAL DEFAULT 0")
        add_column_if_missing(c, "logs", "rerank_method", "TEXT")

        # Feedback table
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER,
                query_id   TEXT,
                thumbs_up  INTEGER,
                timestamp  TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER,
                role       TEXT,
                content    TEXT,
                timestamp  TEXT
            )
        """)

        # Seed default users if they don't exist
        c.execute("SELECT id FROM users WHERE username = 'admin'")
        if not c.fetchone():
            admin_pw = bcrypt.hashpw(b"admin123", bcrypt.gensalt())
            student_pw = bcrypt.hashpw(b"student123", bcrypt.gensalt())
            c.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("admin", admin_pw, "admin"),
            )
            c.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("student", student_pw, "student"),
            )

        conn.commit()
    print("Database initialized.")


def get_user(username: str) -> dict | None:
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, username, password, role FROM users WHERE username = ?",
            (username,),
        )
        row = c.fetchone()
        if row:
            return {"id": row[0], "username": row[1], "password": row[2], "role": row[3]}
        return None


def create_user(username: str, password: str, role: str = "student") -> bool:
    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed, role),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # username already exists


def add_log(
    user_id: int,
    username: str,
    query: str,
    response: str,
    route: str,
    allowed: int,
    query_id: str | None = None,
    collection: str | None = None,
    chunks_found: int = 0,
    top_rerank_score: float = 0.0,
    rerank_method: str | None = None,
):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO logs (
                user_id, username, query, response, route, allowed, timestamp,
                query_id, collection, chunks_found, top_rerank_score, rerank_method
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                user_id, username, query, response, route, allowed, datetime.now().isoformat(),
                query_id, collection, chunks_found, top_rerank_score, rerank_method,
            ),
        )
        conn.commit()


def get_logs_for_admin(limit: int = 500) -> list[dict]:
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT id, username, query, route, allowed, timestamp, query_id,
                   collection, chunks_found, top_rerank_score, rerank_method
            FROM logs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = c.fetchall()
    return [
        {
            "id": r[0],
            "username": r[1],
            "query": r[2],
            "route": r[3],
            "allowed": r[4],
            "timestamp": r[5],
            "query_id": r[6],
            "collection": r[7],
            "chunks_found": r[8] or 0,
            "top_rerank_score": r[9] or 0.0,
            "rerank_method": r[10] or "",
        }
        for r in rows
    ]


def save_feedback(user_id: int, query_id: str, thumbs_up: bool):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, query_id, thumbs_up, timestamp) VALUES (?,?,?,?)",
            (user_id, query_id, 1 if thumbs_up else 0, datetime.now().isoformat()),
        )
        conn.commit()


def get_feedback_stats() -> tuple[int, int]:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM feedback WHERE thumbs_up = 1")
        positive = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM feedback")
        total = c.fetchone()[0]
    return positive, total


def save_memory(user_id: int, role: str, content: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversation_memory (user_id, role, content, timestamp) VALUES (?,?,?,?)",
            (user_id, role, content, datetime.now().isoformat()),
        )
        conn.commit()


def get_recent_memory(user_id: int, limit: int = 8) -> str:
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT role, content
            FROM conversation_memory
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = list(reversed(c.fetchall()))

    return "\n".join(f"{row[0].title()}: {row[1]}" for row in rows)


def get_admin_analytics(limit: int = 500) -> dict:
    logs = get_logs_for_admin(limit)
    positive, total_fb = get_feedback_stats()

    allowed_logs = [log for log in logs if log["allowed"] == 1]
    blocked = len(logs) - len(allowed_logs)
    route_dist = Counter(log["route"] for log in logs)
    collection_dist = Counter(log["collection"] or "none" for log in allowed_logs)
    rerank_methods = Counter(log["rerank_method"] or "unknown" for log in allowed_logs)
    top_questions = Counter(log["query"] for log in allowed_logs).most_common(10)
    active_users = Counter(log["username"] for log in logs).most_common(10)

    daily = defaultdict(lambda: {"total": 0, "blocked": 0})
    for log in logs:
        day = (log["timestamp"] or "")[:10]
        daily[day]["total"] += 1
        if log["allowed"] == 0:
            daily[day]["blocked"] += 1

    avg_chunks = (
        sum(log["chunks_found"] for log in allowed_logs) / len(allowed_logs)
        if allowed_logs else 0.0
    )
    no_context = sum(1 for log in allowed_logs if log["chunks_found"] == 0)
    fb_rate = positive / total_fb if total_fb else 0.0

    return {
        "total_queries": len(logs),
        "blocked_queries": blocked,
        "allowed_queries": len(allowed_logs),
        "route_distribution": dict(route_dist),
        "collection_distribution": dict(collection_dist),
        "rerank_methods": dict(rerank_methods),
        "top_questions": [{"question": q, "count": c} for q, c in top_questions],
        "active_users": [{"username": u, "count": c} for u, c in active_users],
        "daily_queries": [
            {"date": day, "total": values["total"], "blocked": values["blocked"]}
            for day, values in sorted(daily.items())
        ],
        "avg_chunks_found": round(avg_chunks, 2),
        "no_context_queries": no_context,
        "feedback_positive_rate": round(fb_rate * 100, 1),
        "total_feedback": total_fb,
    }


def get_all_users() -> list[dict]:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, role FROM users ORDER BY id")
        rows = c.fetchall()
    return [{"id": r[0], "username": r[1], "role": r[2]} for r in rows]
