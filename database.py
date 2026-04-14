import csv
import datetime as dt
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import config


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT DEFAULT 'present',
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_user_date
            ON attendance(user_id, date)
            """
        )
        conn.commit()


def add_user(name: str) -> int:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO users(name, created_at) VALUES(?, ?)", (name, now))
        conn.commit()
        return int(cur.lastrowid)


def get_user_by_id(user_id: int) -> Optional[Dict]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, created_at FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def get_all_users() -> List[Dict]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, created_at FROM users ORDER BY id DESC")
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def check_attendance_today(user_id: int) -> bool:
    today = dt.date.today().strftime("%Y-%m-%d")
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM attendance WHERE user_id = ? AND date = ? LIMIT 1",
            (user_id, today),
        )
        return cur.fetchone() is not None


def mark_attendance(user_id: int) -> bool:
    if check_attendance_today(user_id):
        return False

    now = dt.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO attendance(user_id, date, time, status) VALUES(?, ?, ?, 'present')",
            (user_id, date_str, time_str),
        )
        conn.commit()

    export_date_csv(date_str)
    return True


def get_attendance_logs(
    date_filter: Optional[str] = None,
    user_filter: Optional[int] = None,
    page: int = 1,
    page_size: int = 20,
) -> Tuple[List[Dict], int]:
    where = []
    params = []

    if date_filter:
        where.append("a.date = ?")
        params.append(date_filter)
    if user_filter:
        where.append("a.user_id = ?")
        params.append(user_filter)

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT COUNT(*) AS total
            FROM attendance a
            JOIN users u ON u.id = a.user_id
            {where_clause}
            """,
            params,
        )
        total = int(cur.fetchone()["total"])

        offset = (max(page, 1) - 1) * page_size
        cur.execute(
            f"""
            SELECT a.id, a.user_id, u.name, a.date, a.time, a.status
            FROM attendance a
            JOIN users u ON u.id = a.user_id
            {where_clause}
            ORDER BY a.date DESC, a.time DESC
            LIMIT ? OFFSET ?
            """,
            [*params, page_size, offset],
        )
        rows = cur.fetchall()

    return [dict(r) for r in rows], total


def get_today_attendance() -> List[Dict]:
    today = dt.date.today().strftime("%Y-%m-%d")
    rows, _ = get_attendance_logs(date_filter=today, page=1, page_size=10_000)
    return rows


def get_attendance_dates() -> List[str]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC")
        rows = cur.fetchall()
    return [str(r["date"]) for r in rows]


def get_csv_attendance_dates(directory: Optional[str] = None) -> List[str]:
    target_dir = directory or config.PROCESSED_DATA_PATH
    if not os.path.exists(target_dir):
        return []

    prefix = "attendance_"
    suffix = ".csv"
    dates = []

    for file_name in os.listdir(target_dir):
        lower = file_name.lower()
        if not (lower.startswith(prefix) and lower.endswith(suffix)):
            continue

        date_str = file_name[len(prefix) : -len(suffix)]
        try:
            dt.datetime.strptime(date_str, "%Y-%m-%d")
            dates.append(date_str)
        except ValueError:
            continue

    return sorted(set(dates), reverse=True)


def get_all_available_attendance_dates(directory: Optional[str] = None) -> List[str]:
    db_dates = set(get_attendance_dates())
    csv_dates = set(get_csv_attendance_dates(directory=directory))
    return sorted(db_dates.union(csv_dates), reverse=True)


def get_attendance_for_date(date_str: str) -> List[Dict]:
    rows, _ = get_attendance_logs(date_filter=date_str, page=1, page_size=100_000)
    return rows


def load_attendance_csv_by_date(date_str: str, directory: Optional[str] = None) -> List[Dict]:
    target_dir = directory or config.PROCESSED_DATA_PATH
    filepath = os.path.join(target_dir, f"attendance_{date_str}.csv")
    if not os.path.exists(filepath):
        return []

    rows: List[Dict] = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "id": row.get("id", ""),
                    "user_id": row.get("user_id", ""),
                    "name": row.get("name", ""),
                    "date": row.get("date", date_str),
                    "time": row.get("time", ""),
                    "status": row.get("status", "present"),
                }
            )
    return rows


def export_date_csv(date_str: str, directory: Optional[str] = None) -> str:
    target_dir = directory or config.PROCESSED_DATA_PATH
    os.makedirs(target_dir, exist_ok=True)

    filepath = os.path.join(target_dir, f"attendance_{date_str}.csv")
    rows = get_attendance_for_date(date_str)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "user_id", "name", "date", "time", "status"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return filepath


def export_all_dates_csv(directory: Optional[str] = None) -> List[str]:
    files = []
    for date_str in get_attendance_dates():
        files.append(export_date_csv(date_str, directory=directory))
    return files


def export_to_csv(filepath: str) -> str:
    rows, _ = get_attendance_logs(page=1, page_size=1_000_000)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "user_id", "name", "date", "time", "status"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return filepath
