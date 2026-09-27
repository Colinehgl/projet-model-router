import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "logs.db"

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            model_used TEXT,
            response TEXT,
            latency_ms REAL,
            estimated_cost REAL
        )
    """)
    conn.commit()
    conn.close()

def log_request(query: str, model_used: str, response: str, latency_ms: float, estimated_cost: float = 0.0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO requests (timestamp, query, model_used, response, latency_ms, estimated_cost)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), query, model_used, response, latency_ms, estimated_cost))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    log_request("test query", "local", "test response", 123.4, 0.0)
    print("Base initialisée et ligne de test insérée avec succès.")