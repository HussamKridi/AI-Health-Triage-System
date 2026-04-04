import sqlite3

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn, table_name, column_name, column_definition):
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    column_names = {column["name"] for column in columns}
    if column_name not in column_names:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS vitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            gender TEXT,
            weight REAL,
            height REAL,
            spo2 REAL,
            temperature REAL,
            heart_rate INTEGER,
            classification TEXT,
            triage_status TEXT DEFAULT 'questioning',
            ai_recommendation TEXT,
            model_assessment TEXT,
            disagreement_logged INTEGER DEFAULT 0,
            final_source TEXT DEFAULT 'pending'
        )
    ''')
    _ensure_column(conn, "vitals", "conversation_history", "TEXT DEFAULT '[]'")
    _ensure_column(conn, "vitals", "age", "INTEGER")
    _ensure_column(conn, "vitals", "gender", "TEXT")
    _ensure_column(conn, "vitals", "weight", "REAL")
    _ensure_column(conn, "vitals", "height", "REAL")
    _ensure_column(conn, "vitals", "triage_status", "TEXT DEFAULT 'questioning'")
    _ensure_column(conn, "vitals", "model_assessment", "TEXT")
    _ensure_column(conn, "vitals", "disagreement_logged", "INTEGER DEFAULT 0")
    _ensure_column(conn, "vitals", "final_source", "TEXT DEFAULT 'pending'")
    conn.commit()
    conn.close()
