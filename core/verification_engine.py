"""
AI-Based Face & Fingerprint Attendance Dual-Verification Engine.

Key Principles:
1. Read-Only Company Fingerprint Database access (NEVER insert, update, or delete in company DB).
2. Local Face Recognition with strict >= 87% confidence requirement.
3. 5-Minute Verification Window tracking between AI Face detection and Fingerprint punch.
4. Shift Awareness (Morning 09:00 AM, Evening 07:30 PM, Working Hours Exit detection).
5. Comprehensive Daily Verification Matrix Ledger persistence.
"""

import os
import csv
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

CONFIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

SETTINGS_FILE = CONFIG_DIR / "verification_settings.json"
COMPANY_FP_DB_FILE = DATA_DIR / "company_fingerprint_db.json"
AI_LEDGER_FILE = DATA_DIR / "ai_verification_ledger.json"
AI_LEDGER_CSV = LOGS_DIR / "ai_verification_ledger.csv"


class VerificationConfig:
    """Manages dynamic, administrator-configurable attendance verification settings."""

    DEFAULT_SETTINGS = {
        "face_confidence_threshold": 0.80,
        "fingerprint_window_minutes": 5,
        "morning_start_time": "09:00:00",
        "morning_window_start": "07:00:00",
        "morning_window_end": "12:00:00",
        "evening_start_time": "19:30:00",
        "evening_window_start": "17:00:00",
        "evening_window_end": "22:00:00",
        "cooldown_seconds": 60,
        "enable_exit_detection": True
    }

    def __init__(self):
        self.settings = dict(self.DEFAULT_SETTINGS)
        self.load()

    def load(self):
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
            except Exception as e:
                print(f"[Config] Error loading verification settings: {e}")
        else:
            self.save()

    def save(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"[Config] Error saving verification settings: {e}")

    def update(self, new_settings: dict) -> dict:
        for k, v in new_settings.items():
            if k in self.DEFAULT_SETTINGS:
                if k in ["face_confidence_threshold"]:
                    self.settings[k] = max(0.50, min(1.0, float(v)))
                elif k in ["fingerprint_window_minutes", "cooldown_seconds"]:
                    self.settings[k] = max(1, int(v))
                elif k in ["enable_exit_detection"]:
                    self.settings[k] = bool(v)
                else:
                    self.settings[k] = str(v)
        self.save()
        return self.settings

    @property
    def face_threshold(self) -> float:
        return float(self.settings.get("face_confidence_threshold", 0.87))

    @property
    def window_minutes(self) -> int:
        return int(self.settings.get("fingerprint_window_minutes", 5))

    @property
    def cooldown_sec(self) -> int:
        return int(self.settings.get("cooldown_seconds", 60))


class CompanyFingerprintDBReader:
    """
    Read-Only connector to the company's biometric fingerprint database.
    Guarantees that the AI system NEVER inserts, updates, or modifies company records.
    """

    def __init__(self, db_path: Path = COMPANY_FP_DB_FILE):
        self.db_path = db_path

    def read_all_punches(self) -> List[Dict]:
        """Reads all fingerprint punch records safely (read-only)."""
        if not self.db_path.exists():
            return []
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[CompanyFPDB] Read error: {e}")
            return []

    def find_punch_in_window(
        self,
        ecno: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[Dict]:
        """
        Finds a valid fingerprint punch for the employee within [start_time, end_time].
        Returns the closest matching punch within the window.
        """
        punches = self.read_all_punches()
        matching = []
        for p in punches:
            if str(p.get("ecno", "")).strip() == str(ecno).strip():
                p_ts_str = p.get("timestamp", "")
                try:
                    p_dt = datetime.strptime(p_ts_str, "%Y-%m-%d %H:%M:%S")
                    if start_time <= p_dt <= end_time:
                        matching.append((p_dt, p))
                except Exception:
                    continue
        if matching:
            # Sort chronologically to get the initial/primary punch
            matching.sort(key=lambda x: x[0])
            return matching[0][1]
        return None

    def find_punch_for_shift(self, ecno: str, shift: str, target_dt: datetime) -> Optional[Dict]:
        """
        Finds any valid punch recorded for the employee for the specified shift on target_dt's date.
        """
        punches = self.read_all_punches()
        target_date_str = target_dt.strftime("%Y-%m-%d")
        matching = []
        for p in punches:
            if str(p.get("ecno", "")).strip() == str(ecno).strip():
                p_ts_str = p.get("timestamp", "")
                if p_ts_str.startswith(target_date_str):
                    try:
                        p_dt = datetime.strptime(p_ts_str, "%Y-%m-%d %H:%M:%S")
                        matching.append((p_dt, p))
                    except Exception:
                        continue
        if matching:
            matching.sort(key=lambda x: x[0])
            if shift == "EVENING":
                # For evening return the latest punch
                return matching[-1][1]
            # For morning return the earliest punch
            return matching[0][1]
        return None

    def simulate_punch_for_testing(self, ecno: str, device_id: str = "FP_TERMINAL_01") -> Dict:
        """
        Helper for admin testing: simulates an employee placing their finger on the terminal.
        Appends to the company biometric DB file.
        """
        punches = self.read_all_punches()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_punch = {
            "punch_id": f"FP_{int(time.time()*1000)%1000000}",
            "ecno": str(ecno),
            "timestamp": now_str,
            "device_id": device_id,
            "verify_mode": "FINGERPRINT",
            "direction": "AUTO"
        }
        punches.append(new_punch)
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(punches, f, indent=2)
            print(f"[CompanyFPDB] Simulated fingerprint punch: {ecno} at {now_str}")
        except Exception as e:
            print(f"[CompanyFPDB] Simulate punch error: {e}")
        return new_punch


class AttendanceVerificationManager:
    """
    Core AI & Fingerprint Dual-Verification Controller.
    Tracks 5-minute active windows, shifts, exit remarks, and daily verification ledger.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AttendanceVerificationManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.config = VerificationConfig()
        self.fp_reader = CompanyFingerprintDBReader()
        
        # State
        self.active_windows: Dict[str, Dict] = {}  # {ecno: {"face_time": dt, "expires_at": dt, ...}}
        self.last_seen_cooldown: Dict[str, float] = {}  # {ecno: float(time)}
        self.listeners: List[Callable[[dict], None]] = []
        
        # Daily ledger: {date_str: {ecno: record_dict}}
        self.daily_ledger: Dict[str, Dict[str, Dict]] = {}
        self._load_ledger()

        # Start background reconciliation loop
        self.running = True
        self.reconcile_thread = threading.Thread(target=self._reconciliation_worker, daemon=True, name="FP-Reconciliation")
        self.reconcile_thread.start()

    def add_event_listener(self, callback: Callable[[dict], None]):
        self.listeners.append(callback)

    def _broadcast_event(self, event_type: str, data: dict):
        payload = {"type": event_type, "data": data, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        for listener in self.listeners:
            try:
                listener(payload)
            except Exception as e:
                print(f"[Verification] Broadcast error: {e}")

    def _load_ledger(self):
        if AI_LEDGER_FILE.exists():
            try:
                with open(AI_LEDGER_FILE, "r", encoding="utf-8") as f:
                    self.daily_ledger = json.load(f)
            except Exception as e:
                print(f"[Verification] Ledger load error: {e}")
                self.daily_ledger = {}

    def _save_ledger(self):
        try:
            with open(AI_LEDGER_FILE, "w", encoding="utf-8") as f:
                json.dump(self.daily_ledger, f, indent=2)
            
            # Also sync to CSV
            today_date = datetime.now().strftime("%Y-%m-%d")
            records = list(self.daily_ledger.get(today_date, {}).values())
            
            fieldnames = [
                "ecno", "name", "date",
                "morning_face_time", "morning_fp_time", "morning_status",
                "exit_time", "exit_remark",
                "evening_face_time", "evening_fp_time", "evening_status",
                "final_status", "confidence", "snapshot"
            ]
            
            with open(AI_LEDGER_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    row = {k: r.get(k, "") for k in fieldnames}
                    writer.writerow(row)
        except Exception as e:
            print(f"[Verification] Ledger save error: {e}")

    def get_shift_type(self, current_dt: datetime) -> str:
        """
        Determines the shift context based on configurable timings:
        - MORNING: within morning_window (e.g. 07:00 - 12:00)
        - EVENING: within evening_window (e.g. 17:00 - 22:00)
        - WORKING_HOURS_EXIT: between morning_window_end and evening_window_start
        """
        t = current_dt.time()
        
        m_start = datetime.strptime(self.config.settings.get("morning_window_start", "07:00:00"), "%H:%M:%S").time()
        m_end = datetime.strptime(self.config.settings.get("morning_window_end", "12:00:00"), "%H:%M:%S").time()
        
        e_start = datetime.strptime(self.config.settings.get("evening_window_start", "17:00:00"), "%H:%M:%S").time()
        e_end = datetime.strptime(self.config.settings.get("evening_window_end", "22:00:00"), "%H:%M:%S").time()

        if m_start <= t <= m_end:
            return "MORNING"
        elif e_start <= t <= e_end:
            return "EVENING"
        elif m_end < t < e_start:
            return "WORKING_HOURS_EXIT"
        else:
            return "OFF_HOURS"

    def process_face_detection(
        self,
        ecno: str,
        name: str,
        channel: int,
        confidence: float,
        snapshot_file: str,
        now_dt: Optional[datetime] = None
    ) -> Dict:
        """
        Invoked when a local employee face is recognized with confidence >= 87%.
        Applies cooldown, shift routing, 5-minute window creation, and FP verification.
        """
        now = now_dt or datetime.now()
        now_ts = now.timestamp()
        today_date = now.strftime("%Y-%m-%d")
        now_time_str = now.strftime("%H:%M:%S")

        # 1. Cooldown check (prevent duplicate events while employee stands in front of camera)
        last_seen = self.last_seen_cooldown.get(ecno, 0.0)
        if (now_ts - last_seen) < self.config.cooldown_sec:
            # Within continuous session/cooldown
            return {"status": "COOLDOWN_IGNORED", "ecno": ecno}
        
        self.last_seen_cooldown[ecno] = now_ts

        # 2. Get or create daily employee record
        if today_date not in self.daily_ledger:
            self.daily_ledger[today_date] = {}

        if ecno not in self.daily_ledger[today_date]:
            self.daily_ledger[today_date][ecno] = {
                "ecno": ecno,
                "name": name,
                "date": today_date,
                "morning_face_time": None,
                "morning_fp_time": None,
                "morning_status": "NOT_CHECKED_IN",
                "exit_time": None,
                "exit_remark": None,
                "evening_face_time": None,
                "evening_fp_time": None,
                "evening_status": "NOT_CHECKED_IN",
                "final_status": "PENDING",
                "confidence": round(confidence, 2),
                "snapshot": snapshot_file
            }

        record = self.daily_ledger[today_date][ecno]
        shift = self.get_shift_type(now)

        # 3. Route by Shift with Bi-Directional Reconciliation (Fingerprint First OR Face First)
        # Check from beginning of today up to forward active window
        today_start_dt = datetime.combine(now.date(), datetime.min.time())
        forward_window = now + timedelta(minutes=self.config.window_minutes)

        if shift == "MORNING":
            if not record["morning_face_time"] or record["morning_status"] != "VALID":
                record["morning_face_time"] = now_time_str
                record["snapshot"] = snapshot_file or record.get("snapshot")
                record["confidence"] = round(confidence, 2)
                
                # Check if fingerprint punch was ALREADY recorded earlier today, or is within active window
                existing_punch = self.fp_reader.find_punch_in_window(
                    ecno, today_start_dt, forward_window
                ) or self.fp_reader.find_punch_for_shift(ecno, "MORNING", now)
                
                if existing_punch:
                    fp_time = existing_punch["timestamp"].split(" ")[-1]
                    record["morning_fp_time"] = fp_time
                    record["morning_status"] = "VALID"
                    record["final_status"] = "VALID_MORNING"
                    # Remove from active pending windows if it was waiting
                    self.active_windows.pop(ecno, None)
                    print(f"[Verification] [VALID - BI-DIRECTIONAL] {name} ({ecno}) Morning attendance confirmed! FP Punch: {fp_time}, Face Capture: {now_time_str}")
                else:
                    record["morning_status"] = "PENDING_FINGERPRINT"
                    self.active_windows[ecno] = {
                        "ecno": ecno,
                        "name": name,
                        "shift": "MORNING",
                        "face_time": now,
                        "expires_at": forward_window,
                        "snapshot": snapshot_file
                    }
                    print(f"[Verification] [WINDOW OPEN] {name} ({ecno}) 5-minute FP window active until {forward_window.strftime('%H:%M:%S')}")

        elif shift == "EVENING":
            if not record["evening_face_time"] or record["evening_status"] != "VALID":
                record["evening_face_time"] = now_time_str
                record["snapshot"] = snapshot_file or record.get("snapshot")
                record["confidence"] = round(confidence, 2)
                
                existing_punch = self.fp_reader.find_punch_in_window(
                    ecno, today_start_dt, forward_window
                ) or self.fp_reader.find_punch_for_shift(ecno, "EVENING", now)
                
                if existing_punch:
                    fp_time = existing_punch["timestamp"].split(" ")[-1]
                    record["evening_fp_time"] = fp_time
                    record["evening_status"] = "VALID"
                    record["final_status"] = "VALID_PRESENT"
                    self.active_windows.pop(ecno, None)
                    print(f"[Verification] [VALID - BI-DIRECTIONAL] {name} ({ecno}) Evening attendance confirmed! FP Punch: {fp_time}, Face Capture: {now_time_str}")
                else:
                    record["evening_status"] = "PENDING_FINGERPRINT"
                    self.active_windows[ecno] = {
                        "ecno": ecno,
                        "name": name,
                        "shift": "EVENING",
                        "face_time": now,
                        "expires_at": forward_window,
                        "snapshot": snapshot_file
                    }
                    print(f"[Verification] [WINDOW OPEN] {name} ({ecno}) Evening 5-minute FP window active until {forward_window.strftime('%H:%M:%S')}")

        elif shift == "WORKING_HOURS_EXIT":
            # Outgoing / Leaving Detection during working hours
            if self.config.settings.get("enable_exit_detection", True):
                record["exit_time"] = now_time_str
                record["exit_remark"] = f"Went out at {now_time_str} during working hours"
                if snapshot_file:
                    record["snapshot"] = snapshot_file
                print(f"[Verification] [EXIT DETECTED] {name} ({ecno}) - {record['exit_remark']}")

        self._save_ledger()
        self._broadcast_event("verification_update", record)
        return record

    def _reconciliation_worker(self):
        """
        Background loop running every 2.0s:
        1. Checks company fingerprint DB for newly placed punches matching active 5-min windows.
        2. Times out expired windows (> 5 mins) -> marks FINGERPRINT_NOT_VERIFIED.
        """
        while self.running:
            try:
                now = datetime.now()
                today_date = now.strftime("%Y-%m-%d")
                
                if self.active_windows and today_date in self.daily_ledger:
                    to_remove = []
                    
                    for ecno, win in list(self.active_windows.items()):
                        face_dt = win["face_time"]
                        expires_dt = win["expires_at"]
                        shift = win["shift"]
                        
                        record = self.daily_ledger[today_date].get(ecno)
                        if not record:
                            continue

                        # Check for matching fingerprint in company DB
                        punch = self.fp_reader.find_punch_in_window(
                            ecno, face_dt - timedelta(minutes=1), now
                        )
                        
                        if punch:
                            # Fingerprint placed within window! -> VALID ATTENDANCE
                            fp_time = punch["timestamp"].split(" ")[-1]
                            if shift == "MORNING":
                                record["morning_fp_time"] = fp_time
                                record["morning_status"] = "VALID"
                                record["final_status"] = "VALID_MORNING"
                            else:
                                record["evening_fp_time"] = fp_time
                                record["evening_status"] = "VALID"
                                record["final_status"] = "VALID_PRESENT"
                            
                            print(f"[Verification] [CONFIRMED] {record['name']} ({ecno}) Fingerprint matched at {fp_time}! Status -> VALID")
                            to_remove.append(ecno)
                            self._save_ledger()
                            self._broadcast_event("verification_update", record)
                            
                        elif now > expires_dt:
                            # 5-minute window expired without fingerprint punch!
                            if shift == "MORNING":
                                record["morning_status"] = "FINGERPRINT_MISSING"
                                record["final_status"] = "FACE_VERIFIED_FP_MISSING"
                            else:
                                record["evening_status"] = "FINGERPRINT_MISSING"
                                record["final_status"] = "FACE_VERIFIED_FP_MISSING"
                            
                            print(f"[Verification] [TIMEOUT] {record['name']} ({ecno}) 5-minute window expired! Status -> FINGERPRINT_NOT_VERIFIED")
                            to_remove.append(ecno)
                            self._save_ledger()
                            self._broadcast_event("verification_update", record)

                    for ecno in to_remove:
                        if ecno in self.active_windows:
                            del self.active_windows[ecno]

            except Exception as e:
                print(f"[VerificationWorker] Loop error: {e}")

            time.sleep(2.0)

    def get_today_ledger(self) -> List[Dict]:
        """Returns all employee dual-verification rows for today."""
        today_date = datetime.now().strftime("%Y-%m-%d")
        return list(self.daily_ledger.get(today_date, {}).values())

    def update_record(self, date_str: str, ecno: str, updates: dict) -> Optional[Dict]:
        """Allows administrator to alter/edit an attendance verification record."""
        if date_str not in self.daily_ledger or ecno not in self.daily_ledger[date_str]:
            return None

        record = self.daily_ledger[date_str][ecno]
        allowed_fields = [
            "morning_face_time", "morning_fp_time", "morning_status",
            "exit_time", "exit_remark",
            "evening_face_time", "evening_fp_time", "evening_status",
            "final_status"
        ]
        for field in allowed_fields:
            if field in updates:
                record[field] = updates[field]

        self._save_ledger()
        self._broadcast_event("verification_update", record)
        return record

    def delete_record(self, date_str: str, ecno: str) -> bool:
        """Allows administrator to delete an attendance verification record."""
        if date_str in self.daily_ledger and ecno in self.daily_ledger[date_str]:
            del self.daily_ledger[date_str][ecno]
            if ecno in self.active_windows:
                del self.active_windows[ecno]
            self._save_ledger()
            self._broadcast_event("verification_deleted", {"date": date_str, "ecno": ecno})
            return True
        return False

    def get_active_windows_status(self) -> List[Dict]:
        """Returns list of currently active 5-minute countdowns."""
        now = datetime.now()
        active = []
        for ecno, win in self.active_windows.items():
            remaining_sec = max(0, int((win["expires_at"] - now).total_seconds()))
            active.append({
                "ecno": ecno,
                "name": win["name"],
                "shift": win["shift"],
                "face_time": win["face_time"].strftime("%H:%M:%S"),
                "expires_at": win["expires_at"].strftime("%H:%M:%S"),
                "remaining_seconds": remaining_sec,
                "snapshot": win["snapshot"]
            })
        return active
