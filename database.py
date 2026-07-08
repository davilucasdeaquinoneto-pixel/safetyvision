import sqlite3


def create_database():

    conn = sqlite3.connect("safetyvision.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            environment TEXT,
            risk_level TEXT,
            risk_detected BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_analysis(filename, environment, analysis):

    conn = sqlite3.connect("safetyvision.db")

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analyses
        (filename, environment, risk_level, risk_detected)
        VALUES (?, ?, ?, ?)
    """,
    (
        filename,
        environment,
        analysis["risk_level"],
        analysis["risk_detected"]
    ))

    conn.commit()
    conn.close()


def get_analyses():

    conn = sqlite3.connect("safetyvision.db")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM analyses
    """)

    analyses = cursor.fetchall()

    conn.close()

    return [dict(row) for row in analyses]


def get_analysis(id):

    conn = sqlite3.connect("safetyvision.db")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM analyses WHERE id = ?
        """,
        (id,)
    )

    analysis = cursor.fetchone()

    conn.close()

    return dict(analysis) if analysis else None