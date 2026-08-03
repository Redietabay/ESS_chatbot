import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = None
try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()
    cur.execute("DELETE FROM query_cache;")
    conn.commit()
    print(f"Cleared {cur.rowcount} cached rows")
    cur.close()
except Exception as e:
    print(f"Failed to clear cache: {e}")
    if conn:
        conn.rollback()
finally:
    if conn:
        conn.close()