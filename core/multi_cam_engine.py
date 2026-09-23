"""
Multi-Camera Real-Time Streaming & AI Attendance Coordinator for Render / Linux.
Manages concurrent Dahua RTSP video channels with YOLOv8 Human + YuNet + SFace Face AI.
"""

import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable
import cv2
import numpy as np

from config.channels_config import get_active_channels, get_channel_name
from config.settings import Config
from core.rtsp_stream import RTSPChannelStream
from core.face_engine import FaceRecognitionEngine
from core.verification_engine import AttendanceVerificationManager

BASE_DIR = Path(__file__).resolve().parent.parent


class MultiCameraAttendanceManager:
    """Manages concurrent Dahua RTSP video feeds on Render with AI face & fingerprint verification."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MultiCameraAttendanceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.channels = get_active_channels()
        self.face_engine = FaceRecognitionEngine()
        self.verification_mgr = AttendanceVerificationManager()

        self.streams: Dict[int, RTSPChannelStream] = {}
        self.worker_threads: Dict[int, threading.Thread] = {}
        self.running = False

        # Metrics
        self.stats = {
            "total_detections": 0,
            "total_matches": 0,
            "start_time": time.time(),
            "active_cams": len(self.channels)
        }

        # Event Listeners
        self.listeners: List[Callable[[dict], None]] = []

    def add_attendance_listener(self, callback: Callable[[dict], None]):
        self.listeners.append(callback)
        self.verification_mgr.add_event_listener(callback)

    def _broadcast_attendance(self, record: dict):
        for cb in self.listeners:
            try:
                cb(record)
            except Exception as e:
                print(f"[MultiCam] Listener error: {e}")

    def start(self):
        if self.running:
            return
        self.running = True

        print(f"[MultiCam] Initializing Dahua RTSP Streams for {len(self.channels)} Channels...")

        for ch in self.channels:
            rtsp_url = Config.get_rtsp_url(ch, subtype=Config.DEFAULT_SUBTYPE)
            stream = RTSPChannelStream(channel=ch, rtsp_url=rtsp_url)
            stream.start()
            self.streams[ch] = stream

            t = threading.Thread(target=self._channel_ai_loop, args=(ch,), daemon=True, name=f"CamAI-CH{ch}")
            self.worker_threads[ch] = t
            t.start()
            print(f"[MultiCam] Started AI streaming worker for Channel {ch:02d} ({get_channel_name(ch)})")

    def _channel_ai_loop(self, channel: int):
        """Worker thread for each camera channel performing real-time AI recognition."""
        stream = self.streams.get(channel)
        if not stream:
            return

        fps_delay = 0.04  # ~25 FPS
        last_process_time = 0
        ai_interval = 0.10  # 10 FPS AI inference
        cached_faces = []

        while self.running:
            raw_frame = stream.get_latest_frame()
            if raw_frame is None:
                time.sleep(0.04)
                continue

            now = time.time()

            # Run AI Face Detection & Recognition
            if now - last_process_time >= ai_interval:
                last_process_time = now
                detected_faces = self.face_engine.detect_and_recognize(raw_frame)
                cached_faces = detected_faces

                if detected_faces:
                    self.stats["total_detections"] += len(detected_faces)

                for face in detected_faces:
                    det_conf = face.get("det_confidence", 0.0)
                    if det_conf < self.face_engine.detector_score_thresh:
                        continue

                    if face["is_matched"] and face["ecno"]:
                        self.stats["total_matches"] += 1
                        
                        snap_file = self.face_engine.save_face_snapshot(
                            raw_frame, face["bbox"], channel, is_matched=True, ecno=face["ecno"]
                        )
                        
                        v_result = self.verification_mgr.process_face_detection(
                            ecno=face["ecno"],
                            name=face["name"],
                            channel=channel,
                            confidence=face["match_score"],
                            snapshot_file=snap_file or ""
                        )

                        if v_result.get("morning_status") == "VALID" or v_result.get("evening_status") == "VALID":
                            is_new, rec = self.face_engine.mark_attendance(
                                ecno=face["ecno"],
                                name=face["name"],
                                channel=channel,
                                confidence=face["match_score"],
                                frame=raw_frame,
                                bbox=face["bbox"]
                            )
                            if is_new:
                                self._broadcast_attendance(rec)

            time.sleep(fps_delay)

    def get_annotated_frame(self, channel: int) -> Optional[np.ndarray]:
        """Returns the latest camera frame with AI bounding boxes and employee names drawn."""
        stream = self.streams.get(channel)
        if not stream:
            return None

        raw_frame = stream.get_latest_frame()
        if raw_frame is None:
            return None

        display_frame = raw_frame.copy()
        try:
            detected_faces = self.face_engine.detect_and_recognize(raw_frame)
            for face in detected_faces:
                x, y, w, h = face["bbox"]
                name = face["name"]
                ecno = face.get("ecno")
                is_matched = face["is_matched"]
                score = face.get("match_score", 0.0)

                if is_matched:
                    color = (0, 230, 115)  # Emerald
                    label = f"{name} ({ecno}) {int(score*100)}%"
                else:
                    color = (0, 165, 255)  # Amber
                    label = "Unregistered Person"

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(display_frame, (x, y - 24), (x + len(label) * 9 + 10, y), color, -1)
                cv2.putText(display_frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        except Exception:
            pass

        return display_frame

    def get_latest_jpeg(self, channel: int) -> Optional[bytes]:
        frame = self.get_annotated_frame(channel)
        if frame is None:
            return None
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            return jpeg.tobytes()
        return None

    def get_metrics(self) -> dict:
        uptime = int(time.time() - self.stats["start_time"])
        return {
            "uptime_seconds": uptime,
            "total_detections": self.stats["total_detections"],
            "total_matches": self.stats["total_matches"],
            "active_cameras": len([s for s in self.streams.values() if s.is_connected])
        }

    def stop(self):
        self.running = False
        for s in self.streams.values():
            s.stop()
        self.verification_mgr.stop()
        print("[MultiCam] Stopped all camera streams on Render.")
