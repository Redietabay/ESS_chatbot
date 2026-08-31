import psycopg2
from dotenv import load_dotenv
import os

# .env read file
load_dotenv() 

def create_database_tables():
    try:
        # ከዳታቤዙ ጋር መገናኘት  
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"), 
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        cursor = connection.cursor() 
        
        # የሰንጠረዦቹ SQL ኮድ (የተሻሻለው መዋቅር)
        sql_query = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(256) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS chat_sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(255) DEFAULT 'New Chat',
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
            sender VARCHAR(10) NOT NULL CHECK (sender IN ('user', 'bot')),
            message TEXT NOT NULL,
            source_doc VARCHAR(255),
            source_page INTEGER,
            route_used VARCHAR(20),
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS query_cache (
            id SERIAL PRIMARY KEY,
            question_hash VARCHAR(64) UNIQUE NOT NULL,
            answer TEXT NOT NULL,
            source_doc VARCHAR(255),
            source_page INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) UNIQUE NOT NULL,
            chunk_count INTEGER NOT NULL,
            indexed_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        print("ሰንጠረዦቹን በመፍጠር ላይ... እባክህ ጠብቅ...")
        cursor.execute(sql_query)
        connection.commit()
        
        print(" ማረጋገጫ፦ ሁሉም 5 ሰንጠረዦች በስኬት ተፈጥረዋል! Day 2 ተጠናቋል።")
        
        cursor.close()
        connection.close()

    except Exception as error:
        print(f" ስህተት አጋጥሟል፦ {error}")

if __name__ == "__main__":
    create_database_tables()