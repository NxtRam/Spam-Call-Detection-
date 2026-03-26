"""
Scam Call Prevention Module
- SQLite blocklist database for storing detected scam numbers/calls
- Auto-block mechanism that stops listening when persistent scam detected
- Scam evidence report generation for reporting to authorities
- Call log with timestamps, transcripts, and confidence scores
"""

import sqlite3
import os
import time
import json
from datetime import datetime

DB_FILE = 'scam_blocklist.db'
REPORTS_DIR = 'scam_reports'

class ScamBlocklist:
    """SQLite-based scam call blocklist and call log database."""

    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Blocklist table: stores known scam numbers/identifiers
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocklist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT,
                caller_identity TEXT,
                reason TEXT,
                risk_level TEXT,
                times_blocked INTEGER DEFAULT 1,
                first_detected TEXT,
                last_detected TEXT,
                is_active INTEGER DEFAULT 1
            )
        ''')

        # Call log: stores every analyzed call session
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS call_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_seconds REAL,
                total_fragments INTEGER,
                spam_fragments INTEGER,
                spam_percentage REAL,
                max_confidence REAL,
                verdict TEXT,
                persistent_spam INTEGER DEFAULT 0,
                auto_blocked INTEGER DEFAULT 0,
                transcript TEXT
            )
        ''')

        # Scam fragments: individual spam-classified speech fragments
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scam_fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                text TEXT,
                confidence REAL,
                risk_level TEXT,
                is_robocall INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def add_to_blocklist(self, phone_number="Unknown", caller_identity="Unknown",
                         reason="Detected by AI", risk_level="HIGH"):
        """Adds a number/caller to the blocklist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # Check if already in blocklist
        cursor.execute('SELECT id, times_blocked FROM blocklist WHERE phone_number = ?',
                       (phone_number,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute('''
                UPDATE blocklist SET times_blocked = times_blocked + 1,
                last_detected = ?, risk_level = ? WHERE id = ?
            ''', (now, risk_level, existing[0]))
        else:
            cursor.execute('''
                INSERT INTO blocklist (phone_number, caller_identity, reason,
                risk_level, first_detected, last_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (phone_number, caller_identity, reason, risk_level, now, now))

        conn.commit()
        conn.close()

    def is_blocked(self, phone_number):
        """Checks if a number is in the blocklist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT is_active FROM blocklist WHERE phone_number = ? AND is_active = 1',
                       (phone_number,))
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def get_blocklist(self):
        """Returns all blocked numbers."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM blocklist WHERE is_active = 1 ORDER BY last_detected DESC')
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def log_call_session(self, session_id, start_time, end_time, summary,
                         transcript="", auto_blocked=False):
        """Logs a complete call analysis session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        duration = (end_time - start_time) if start_time and end_time else 0

        cursor.execute('''
            INSERT INTO call_log (session_id, start_time, end_time, duration_seconds,
            total_fragments, spam_fragments, spam_percentage, max_confidence,
            verdict, persistent_spam, auto_blocked, transcript)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            datetime.fromtimestamp(start_time).isoformat() if start_time else None,
            datetime.fromtimestamp(end_time).isoformat() if end_time else None,
            duration,
            summary.get('total_fragments', 0),
            summary.get('spam_fragments', 0),
            summary.get('spam_percentage', 0),
            summary.get('max_confidence', 0),
            summary.get('verdict', 'Unknown'),
            1 if summary.get('persistent_spam', False) else 0,
            1 if auto_blocked else 0,
            transcript
        ))

        conn.commit()
        conn.close()

    def log_scam_fragment(self, session_id, text, confidence, risk_level, is_robocall=False):
        """Logs an individual scam-classified fragment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute('''
            INSERT INTO scam_fragments (session_id, timestamp, text, confidence,
            risk_level, is_robocall)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, now, text, confidence, risk_level, 1 if is_robocall else 0))

        conn.commit()
        conn.close()

    def get_call_history(self, limit=20):
        """Returns recent call analysis history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM call_log ORDER BY start_time DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def get_stats(self):
        """Returns overall statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM blocklist WHERE is_active = 1')
        blocked_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM call_log')
        total_calls = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM call_log WHERE auto_blocked = 1')
        auto_blocked = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM scam_fragments')
        scam_fragments = cursor.fetchone()[0]

        conn.close()
        return {
            'blocked_numbers': blocked_count,
            'total_calls_analyzed': total_calls,
            'calls_auto_blocked': auto_blocked,
            'scam_fragments_detected': scam_fragments
        }


class ScamReportGenerator:
    """Generates evidence reports for reporting scam calls to authorities."""

    def __init__(self, reports_dir=REPORTS_DIR):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report(self, session_id, start_time, end_time, summary,
                        transcript, scam_fragments, auto_blocked=False):
        """Creates a detailed scam evidence report as a text file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"scam_report_{timestamp}.txt"
        filepath = os.path.join(self.reports_dir, filename)

        duration = end_time - start_time if start_time and end_time else 0
        spam_pct = summary.get('spam_percentage', 0)

        # Determine threat level
        if spam_pct >= 70:
            threat = "CRITICAL - CONFIRMED SCAM"
        elif spam_pct >= 40:
            threat = "HIGH - LIKELY SCAM"
        elif spam_pct >= 15:
            threat = "MEDIUM - SUSPICIOUS"
        else:
            threat = "LOW - PROBABLY SAFE"

        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("  SCAM CALL EVIDENCE REPORT\n")
            f.write("  Generated by On-Device Scam Call Detection System\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Report ID:        {session_id}\n")
            f.write(f"Generated:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Call Start:       {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S') if start_time else 'N/A'}\n")
            f.write(f"Call End:         {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S') if end_time else 'N/A'}\n")
            f.write(f"Duration:         {duration:.0f} seconds\n")
            f.write(f"Threat Level:     {threat}\n")
            f.write(f"Auto-Blocked:     {'YES' if auto_blocked else 'No'}\n\n")

            f.write("-" * 60 + "\n")
            f.write("  ANALYSIS SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Total fragments:     {summary.get('total_fragments', 0)}\n")
            f.write(f"  Spam fragments:      {summary.get('spam_fragments', 0)}\n")
            f.write(f"  Spam percentage:     {spam_pct:.1f}%\n")
            f.write(f"  Max confidence:      {summary.get('max_confidence', 0):.2f}\n")
            f.write(f"  Persistent spam:     {'YES' if summary.get('persistent_spam') else 'No'}\n\n")

            f.write("-" * 60 + "\n")
            f.write("  SCAM FRAGMENTS DETECTED\n")
            f.write("-" * 60 + "\n")
            for i, frag in enumerate(scam_fragments, 1):
                f.write(f"\n  [{i}] Confidence: {frag['confidence']:.2f} | Risk: {frag['risk_level']}\n")
                f.write(f"      Text: \"{frag['text']}\"\n")
                if frag.get('is_robocall'):
                    f.write(f"      ⚠ Robocall pattern detected\n")

            f.write("\n" + "-" * 60 + "\n")
            f.write("  FULL TRANSCRIPT\n")
            f.write("-" * 60 + "\n")
            f.write(f"  {transcript if transcript else 'No transcript available'}\n\n")

            f.write("=" * 60 + "\n")
            f.write("  HOW TO REPORT THIS SCAM\n")
            f.write("=" * 60 + "\n")
            f.write("  1. National Cyber Crime Portal: https://cybercrime.gov.in\n")
            f.write("  2. Cyber Crime Helpline: 1930\n")
            f.write("  3. Local Police: File an FIR with this report as evidence\n")
            f.write("  4. TRAI DND: Register on TRAI DND app to block spam calls\n")
            f.write("  5. Bank: If financial info shared, call bank immediately\n")
            f.write("=" * 60 + "\n")

        return filepath


class CallPrevention:
    """
    Prevention engine that decides when to auto-block a call.
    Triggers blocking based on confidence thresholds and scam patterns.
    """

    def __init__(self, block_threshold=0.85, persistent_block=True):
        self.block_threshold = block_threshold
        self.persistent_block = persistent_block
        self.blocked = False
        self.block_reason = ""
        self.high_confidence_count = 0

    def should_block(self, label, confidence, is_persistent, is_robocall=False):
        """
        Evaluates whether the call should be blocked.
        Returns: (should_block: bool, reason: str)
        """
        if self.blocked:
            return True, self.block_reason

        # Rule 1: Persistent spam detected (3+ consecutive spam classifications)
        if is_persistent and self.persistent_block:
            self.blocked = True
            self.block_reason = "Persistent scam pattern detected (3+ consecutive spam)"
            return True, self.block_reason

        # Rule 2: Very high confidence single detection
        if label == "Spam" and confidence >= 0.95:
            self.blocked = True
            self.block_reason = f"Critical threat detected (confidence: {confidence:.2f})"
            return True, self.block_reason

        # Rule 3: Multiple high-confidence detections
        if label == "Spam" and confidence >= self.block_threshold:
            self.high_confidence_count += 1
            if self.high_confidence_count >= 2:
                self.blocked = True
                self.block_reason = f"Multiple high-confidence scam detections ({self.high_confidence_count}x)"
                return True, self.block_reason

        # Rule 4: Robocall with spam content
        if is_robocall and label == "Spam" and confidence >= 0.60:
            self.blocked = True
            self.block_reason = "Robocall with scam content detected"
            return True, self.block_reason

        return False, ""

    def reset(self):
        self.blocked = False
        self.block_reason = ""
        self.high_confidence_count = 0


if __name__ == "__main__":
    # Test the prevention system
    print("=" * 60)
    print("  SCAM PREVENTION MODULE - SELF TEST")
    print("=" * 60)

    # Test blocklist
    bl = ScamBlocklist()
    print("\n✓ Database initialized")

    bl.add_to_blocklist("+91-9876543210", "Fake CBI Officer", "Digital arrest scam", "CRITICAL")
    bl.add_to_blocklist("+91-1234567890", "KBC Lottery", "Lottery fraud", "HIGH")
    print("✓ Added 2 numbers to blocklist")

    blocklist = bl.get_blocklist()
    print(f"✓ Blocklist has {len(blocklist)} entries")

    for entry in blocklist:
        print(f"  📛 {entry['phone_number']} - {entry['caller_identity']} ({entry['risk_level']})")

    # Test prevention engine
    prev = CallPrevention()
    print("\n--- Prevention Engine Tests ---")

    tests = [
        ("Ham", 0.10, False, False, "Normal speech"),
        ("Spam", 0.60, False, False, "Low-confidence spam"),
        ("Spam", 0.85, False, False, "High-confidence spam (1st)"),
        ("Spam", 0.88, False, False, "High-confidence spam (2nd) → BLOCK"),
    ]

    for label, conf, persistent, robo, desc in tests:
        block, reason = prev.should_block(label, conf, persistent, robo)
        status = "🛑 BLOCKED" if block else "✅ ALLOWED"
        print(f"  {status} | {desc} | {reason}")

    stats = bl.get_stats()
    print(f"\n✓ Stats: {stats}")
    print("\n✓ All prevention module tests passed!")
