"""
Attendance Manager: Handles SQLite database & CSV logging, session duration tracking,
cooldown deduplication, and analytics reporting.
"""

import csv
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from core.config import ATTENDANCE_CSV, ATTENDANCE_DB, COOLDOWN_SECONDS


class AttendanceManager:
    def __init__(self, csv_path=ATTENDANCE_CSV, db_path=ATTENDANCE_DB, cooldown=COOLDOWN_SECONDS):
        self.csv_path = csv_path
        self.db_path = db_path
        self.cooldown = cooldown

        # In-memory tracking for active camera session:
        # {name: {"entry": datetime, "exit": datetime, "last_logged": datetime}}
        self.active_sessions = {}

        self._init_db()
        self._init_csv()

    def _init_db(self):
        """Initialize SQLite database table with proper schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    duration_hours REAL NOT NULL,
                    status TEXT DEFAULT 'Present',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, date)
                )
            """)
            conn.commit()

    def _init_csv(self):
        """Ensure CSV exists with standard headers and imports any legacy rows."""
        headers = ["Name", "Date", "Entry_Time", "Exit_Time", "Duration_Hours", "Status"]

        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            return

        # Check if file has headers
        with open(self.csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        if "Name" not in first_line:
            # Legacy file without headers: read existing rows, rewrite with headers
            rows = []
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for r in reader:
                    if r:
                        # If legacy row has 5 items (Name, Date, In, Out, Dur)
                        if len(r) == 5:
                            r.append("Present")
                        rows.append(r)

            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

            # Sync legacy rows to DB
            self._sync_csv_to_db()

    def _sync_csv_to_db(self):
        """Syncs rows from CSV to SQLite database to keep both in sync."""
        try:
            df = pd.read_csv(self.csv_path)
            with sqlite3.connect(self.db_path) as conn:
                for _, row in df.iterrows():
                    name = str(row.get("Name", "")).strip()
                    date_val = str(row.get("Date", "")).strip()
                    entry_t = str(row.get("Entry_Time", "")).strip()
                    exit_t = str(row.get("Exit_Time", "")).strip()
                    duration = float(row.get("Duration_Hours", 0.0) or 0.0)
                    status = str(row.get("Status", "Present")).strip()

                    if name and date_val:
                        conn.execute("""
                            INSERT OR REPLACE INTO attendance 
                            (name, date, entry_time, exit_time, duration_hours, status)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (name, date_val, entry_t, exit_t, duration, status))
                conn.commit()
        except Exception:
            pass

    def mark_attendance(self, name):
        """
        Marks or updates attendance for recognized person.
        Returns a dict:
            {
                "status": "new" | "updated" | "cooldown",
                "message": str,
                "entry_time": str,
                "exit_time": str,
                "duration": float
            }
        """
        if not name or name.lower() == "unknown":
            return {"status": "ignored", "message": "Unknown face ignored"}

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        now_str = now.strftime("%H:%M:%S")

        is_new_session = name not in self.active_sessions
        session = self.active_sessions.get(name)

        if is_new_session:
            # Check if this person was already recorded in DB today
            existing = self.get_record_today(name)
            if existing:
                try:
                    entry_dt = datetime.strptime(f"{today} {existing['entry_time']}", "%Y-%m-%d %H:%M:%S")
                except Exception:
                    entry_dt = now
            else:
                entry_dt = now

            self.active_sessions[name] = {
                "entry": entry_dt,
                "exit": now,
                "last_prompt": now
            }
            duration_hrs = (now - entry_dt).total_seconds() / 3600.0

            # Commit to DB & CSV
            self._save_record(name, today, entry_dt.strftime("%H:%M:%S"), now_str, duration_hrs)
            return {
                "status": "new",
                "message": f"Welcome {name}! Checked in at {now_str}.",
                "entry_time": entry_dt.strftime("%H:%M:%S"),
                "exit_time": now_str,
                "duration": round(duration_hrs, 2)
            }

        # Existing session: update exit time
        entry_dt = session["entry"]
        duration_hrs = (now - entry_dt).total_seconds() / 3600.0
        session["exit"] = now

        # Update DB & CSV silently
        self._save_record(name, today, entry_dt.strftime("%H:%M:%S"), now_str, duration_hrs)

        # Notify only if cooldown elapsed since last notification
        time_since_last_prompt = (now - session["last_prompt"]).total_seconds()
        if time_since_last_prompt >= self.cooldown:
            session["last_prompt"] = now
            return {
                "status": "updated",
                "message": f"{name} active. Last seen: {now_str} (Duration: {duration_hrs:.2f}h).",
                "entry_time": entry_dt.strftime("%H:%M:%S"),
                "exit_time": now_str,
                "duration": round(duration_hrs, 2)
            }

        return {
            "status": "cooldown",
            "message": f"{name} detected.",
            "entry_time": entry_dt.strftime("%H:%M:%S"),
            "exit_time": now_str,
            "duration": round(duration_hrs, 2)
        }

    def _save_record(self, name, date_str, entry_time, exit_time, duration_hours):
        """Persist record to SQLite DB and sync to CSV."""
        duration_rounded = round(duration_hours, 2)

        # 1. Update SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO attendance 
                (name, date, entry_time, exit_time, duration_hours, status)
                VALUES (?, ?, ?, ?, ?, 'Present')
            """, (name, date_str, entry_time, exit_time, duration_rounded))
            conn.commit()

        # 2. Sync to CSV
        self._export_db_to_csv()

    def _export_db_to_csv(self):
        """Exports all database rows cleanly to attendance.csv."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT name AS Name, 
                       date AS Date, 
                       entry_time AS Entry_Time, 
                       exit_time AS Exit_Time, 
                       duration_hours AS Duration_Hours, 
                       status AS Status
                FROM attendance
                ORDER BY date DESC, entry_time DESC
            """, conn)
        df.to_csv(self.csv_path, index=False)

    def get_record_today(self, name):
        """Checks if a user has an attendance record today."""
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entry_time, exit_time, duration_hours, status
                FROM attendance
                WHERE name = ? AND date = ?
            """, (name, today))
            row = cursor.fetchone()
            if row:
                return {
                    "entry_time": row[0],
                    "exit_time": row[1],
                    "duration_hours": row[2],
                    "status": row[3]
                }
        return None

    def get_all_records(self, date=None, name=None):
        """Returns attendance DataFrame with optional filtering."""
        query = "SELECT name AS Name, date AS Date, entry_time AS Entry_Time, exit_time AS Exit_Time, duration_hours AS Duration_Hours, status AS Status FROM attendance WHERE 1=1"
        params = []

        if date:
            query += " AND date = ?"
            params.append(date)
        if name:
            query += " AND LOWER(name) LIKE ?"
            params.append(f"%{name.lower()}%")

        query += " ORDER BY date DESC, entry_time DESC"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_today_summary(self):
        """Returns statistics for today's attendance."""
        today = datetime.now().strftime("%Y-%m-%d")
        df_today = self.get_all_records(date=today)
        return {
            "date": today,
            "total_present": len(df_today),
            "attendees": df_today["Name"].tolist() if not df_today.empty else [],
            "records": df_today
        }
