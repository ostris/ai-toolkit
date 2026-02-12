#!/usr/bin/env python3
# jobs.py — Manage AI-Toolkit jobs (list, stop, kill, delete, hung)

import os
import sys
import sqlite3
from datetime import datetime
from uuid import UUID

DB_PATH = "./aitk_db.db"
TABLE = "Job"  # capital J per schema


# ---------- Helpers ----------

def safe_time(val):
    """Safely format timestamp or return raw if invalid."""
    if val is None:
        return ""
    try:
        if isinstance(val, str) and "-" in val and ":" in val:
            return val.strip()
        if isinstance(val, (int, float)) and 0 < val < 1e12:
            return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(val, str) and val.isdigit():
            try:
                return datetime.strptime(val, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return val
        return str(val)
    except Exception:
        return str(val)


def valid_uuid(value: str) -> str:
    """Validate UUID format (returns the original string if valid)."""
    try:
        UUID(value)
        return value
    except ValueError:
        raise ValueError(f"Invalid job UUID: {value}")


def connect_db():
    if not os.path.exists(DB_PATH):
        sys.exit(f"Database not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


# ---------- Core Operations ----------

def list_jobs(filter_hung=False):
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = f"SELECT id, name, status, stop, created_at, updated_at FROM {TABLE}"
    if filter_hung:
        query += " WHERE stop=1 OR status IN ('running', 'stopping')"
    query += " ORDER BY updated_at DESC"

    cur.execute(query)
    rows = cur.fetchall()

    if not rows:
        print("No jobs found in the database." if not filter_hung else "No hung or active jobs found.")
        return

    print(f"{'ID':36} | {'NAME':20} | {'STATUS':10} | {'STOP':4} | {'UPDATED'}")
    print("-" * 90)
    for r in rows:
        print(f"{r['id']:<36} | {r['name']:<20} | {r['status']:<10} | {r['stop']:<4} | {safe_time(r['updated_at'])}")
    conn.close()


def stop_job(job_id):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(f"UPDATE {TABLE} SET stop=1, status='stopping', updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
    conn.commit()
    print(f"Job {job_id} marked as stopping (stop=1, status='stopping').")
    conn.close()


def kill_job(job_id):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(f"UPDATE {TABLE} SET stop=0, status='completed', updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
    conn.commit()
    print(f"Job {job_id} forcibly marked as completed (stop=0, status='completed').")
    conn.close()


def delete_job(job_id):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {TABLE} WHERE id=?", (job_id,))
    conn.commit()
    print(f"Job {job_id} permanently deleted from database.")
    conn.close()


# ---------- Main Entry ----------

def usage():
    print(
        "Usage:\n"
        "  python jobs.py                 → list all jobs\n"
        "  python jobs.py --hung          → list only stuck/running jobs\n"
        "  python jobs.py --stop <uuid>   → mark a job as stopping\n"
        "  python jobs.py --kill <uuid>   → forcibly complete a job\n"
        "  python jobs.py --delete <uuid> → remove a job from the database\n"
    )


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        list_jobs()
        sys.exit(0)

    if len(args) == 1 and args[0] == "--hung":
        list_jobs(filter_hung=True)
        sys.exit(0)

    if len(args) != 2:
        usage()
        sys.exit(1)

    mode, job_id = args
    try:
        job_id = valid_uuid(job_id)
    except ValueError as e:
        sys.exit(str(e))

    if mode == "--stop":
        stop_job(job_id)
    elif mode == "--kill":
        kill_job(job_id)
    elif mode == "--delete":
        delete_job(job_id)
    else:
        usage()
