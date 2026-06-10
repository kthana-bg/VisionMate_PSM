import sqlite3
import json
import pandas as pd
from datetime import datetime
import uuid
import os
import threading

class DatabaseManager:
    
    def __init__(self, db_path: str = "visionmate.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._create_tables()
    
    def _get_connection(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _get_cursor(self):
        conn = self._get_connection()
        return conn.cursor()
    
    def _create_tables(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_name TEXT NOT NULL,
                face_embedding TEXT NOT NULL,
                created_date TEXT NOT NULL,
                last_login TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_start TEXT NOT NULL,
                session_end TEXT,
                session_duration_minutes INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                
                model_a1_score REAL,
                model_a1_class TEXT,
                model_a1_latency REAL,
                
                model_b1_score REAL,
                model_b1_class TEXT,
                model_b1_latency REAL,
                
                model_c1_score REAL,
                model_c1_class TEXT,
                model_c1_latency REAL,
                
                model_a2_score REAL,
                model_a2_class TEXT,
                model_a2_latency REAL,
                
                model_b2_score REAL,
                model_b2_class TEXT,
                model_b2_latency REAL,
                
                model_c2_score REAL,
                model_c2_class TEXT,
                model_c2_latency REAL,
                
                eye_consensus TEXT,
                posture_consensus TEXT,
                health_score INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_user(self, user_name: str, face_embedding: list) -> str:
        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO users (user_id, user_name, face_embedding, created_date, last_login)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, user_name, json.dumps(face_embedding), now, now))
        self._get_connection().commit()
        
        return user_id
    
    def get_all_users(self) -> list:
        # Get all registered users
        cursor = self._get_cursor()
        cursor.execute("SELECT user_id, user_name, face_embedding FROM users")
        rows = cursor.fetchall()
        
        return [{'user_id': row[0], 'user_name': row[1], 'face_embedding': json.loads(row[2])} 
                for row in rows]
    
    def get_user_by_id(self, user_id: str) -> dict:
        # Get user by ID
        cursor = self._get_cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def start_session(self, user_id: str) -> str:
        # Start a new user session
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, user_id, session_start)
            VALUES (?, ?, ?)
        """, (session_id, user_id, now))
        self._get_connection().commit()
        
        return session_id
    
    def end_session(self, session_id: str):
        # End the current session
        now = datetime.now().isoformat()
        
        cursor = self._get_cursor()
        cursor.execute("SELECT session_start FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            start_time = datetime.fromisoformat(row[0])
            end_time = datetime.fromisoformat(now)
            duration = int((end_time - start_time).total_seconds() / 60)
            
            cursor.execute("""
                UPDATE sessions 
                SET session_end = ?, session_duration_minutes = ?
                WHERE session_id = ?
            """, (now, duration, session_id))
            self._get_connection().commit()
    
    def get_current_session(self, session_id: str) -> dict:
        # Get current session information
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT session_start, session_duration_minutes 
            FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        
        if row:
            return {'session_start': row[0], 'duration_minutes': row[1] or 0}
        return {}
    
    def log_model_comparison(self, session_id: str, user_id: str, results: dict):
        # Log model comparison results to database
        eye_results = results.get('eye', {})
        posture_results = results.get('posture', {})
        
        a1 = eye_results.get('A1', {})
        b1 = eye_results.get('B1', {})
        c1 = eye_results.get('C1', {})
        a2 = posture_results.get('A2', {})
        b2 = posture_results.get('B2', {})
        c2 = posture_results.get('C2', {})
        
        cursor = self._get_cursor()
        cursor.execute("""
            INSERT INTO model_logs (
                session_id, user_id, timestamp,
                model_a1_score, model_a1_class, model_a1_latency,
                model_b1_score, model_b1_class, model_b1_latency,
                model_c1_score, model_c1_class, model_c1_latency,
                model_a2_score, model_a2_class, model_a2_latency,
                model_b2_score, model_b2_class, model_b2_latency,
                model_c2_score, model_c2_class, model_c2_latency,
                eye_consensus, posture_consensus, health_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, user_id, datetime.now().isoformat(),
            a1.get('fatigue_score', 0), a1.get('classification', 'NORMAL'), a1.get('latency_ms', 0),
            b1.get('fatigue_score', 0), b1.get('classification', 'NORMAL'), b1.get('latency_ms', 0),
            c1.get('fatigue_score', 0), c1.get('classification', 'NORMAL'), c1.get('latency_ms', 0),
            a2.get('posture_score', 0), a2.get('status', 'GOOD'), a2.get('latency_ms', 0),
            b2.get('posture_score', 0), b2.get('status', 'GOOD'), b2.get('latency_ms', 0),
            c2.get('slouching_prob', 0), c2.get('status', 'GOOD'), c2.get('latency_ms', 0),
            results.get('eye_consensus', 'NORMAL'),
            results.get('posture_consensus', 'GOOD'),
            results.get('health_score', 50)
        ))
        self._get_connection().commit()
    
    def get_average_latencies(self, session_id: str) -> dict:
        # Get average latency for each model
        cursor = self._get_cursor()
        cursor.execute("""
            SELECT 
                AVG(model_a1_latency) as a1,
                AVG(model_b1_latency) as b1,
                AVG(model_c1_latency) as c1,
                AVG(model_a2_latency) as a2,
                AVG(model_b2_latency) as b2,
                AVG(model_c2_latency) as c2
            FROM model_logs WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        
        return {
            'A1': row[0] or 0,
            'B1': row[1] or 0,
            'C1': row[2] or 0,
            'A2': row[3] or 0,
            'B2': row[4] or 0,
            'C2': row[5] or 0
        }
    
    def get_user_analytics(self, user_id: str, hours: int = 24) -> pd.DataFrame:
        # Get user analytics for the last N hours
        query = """
            SELECT 
                timestamp,
                model_c1_score as eye_score,
                model_c1_class as eye_class,
                model_c2_score as posture_score,
                model_c2_class as posture_class,
                health_score
            FROM model_logs
            WHERE user_id = ? 
            AND timestamp >= datetime('now', ?)
            ORDER BY timestamp
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        df = pd.read_sql_query(query, conn, params=(user_id, f'-{hours} hours'))
        conn.close()
        return df
    
    def get_strain_statistics(self, user_id: str, hours: int = 24) -> dict:
        # Get strain detection statistics
        df = self.get_user_analytics(user_id, hours)
        
        if df.empty:
            return {
                'eye_strain_count': 0,
                'posture_poor_count': 0,
                'total_logs': 0,
                'eye_strain_percentage': 0,
                'posture_poor_percentage': 0
            }
        
        eye_strain = (df['eye_class'] == 'STRAINED').sum()
        posture_poor = (df['posture_class'] == 'SLOUCHING').sum()
        
        return {
            'eye_strain_count': int(eye_strain),
            'posture_poor_count': int(posture_poor),
            'total_logs': len(df),
            'eye_strain_percentage': (eye_strain / len(df)) * 100 if len(df) > 0 else 0,
            'posture_poor_percentage': (posture_poor / len(df)) * 100 if len(df) > 0 else 0
        }
    
    def get_session_comparison_data(self, session_id: str) -> pd.DataFrame:
        # Get comparison data for current session
        query = """
            SELECT 
                timestamp,
                model_a1_class, model_b1_class, model_c1_class,
                model_a2_class, model_b2_class, model_c2_class,
                model_c1_latency, model_c2_latency,
                health_score
            FROM model_logs WHERE session_id = ?
            ORDER BY timestamp
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        df = pd.read_sql_query(query, conn, params=(session_id,))
        conn.close()
        return df
    
    def export_all_logs(self, user_id: str) -> pd.DataFrame:
        # Export all logs for a user
        query = """
            SELECT 
                timestamp,
                model_a1_score, model_a1_class, model_a1_latency,
                model_b1_score, model_b1_class, model_b1_latency,
                model_c1_score, model_c1_class, model_c1_latency,
                model_a2_score, model_a2_class, model_a2_latency,
                model_b2_score, model_b2_class, model_b2_latency,
                model_c2_score, model_c2_class, model_c2_latency,
                eye_consensus, posture_consensus, health_score
            FROM model_logs WHERE user_id = ?
            ORDER BY timestamp
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        return df
    
    def close(self):
        # Close all database connections
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
