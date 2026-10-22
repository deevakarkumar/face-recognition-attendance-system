"""
Script 4: View and filter attendance records.
Run in VS Code or Terminal:
    python scripts/4_view_records.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.attendance_manager import AttendanceManager


def main():
    print("=" * 60)
    print("      SMART ATTENDANCE SYSTEM - RECORDS VIEWER")
    print("=" * 60)

    mgr = AttendanceManager()
    summary = mgr.get_today_summary()

    print(f"\n[SUMMARY FOR TODAY: {summary['date']}]")
    print(f"Total Present: {summary['total_present']}")
    if summary['attendees']:
        print(f"Attendees: {', '.join(summary['attendees'])}")
    else:
        print("No attendees recorded yet today.")

    print("\n" + "-" * 60)
    df = mgr.get_all_records()

    if df.empty:
        print("[INFO] No attendance records found in database.")
    else:
        print(f"\n[ALL ATTENDANCE RECORDS ({len(df)} total)]\n")
        print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"Records stored in: {mgr.csv_path}")
    print(f"Database stored in: {mgr.db_path}")


if __name__ == "__main__":
    main()
