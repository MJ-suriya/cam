"""
Robust Multi-Threaded RTSP Stream Worker for Linux / Render Docker.
Connects to Dahua RTSP streams with automatic reconnection and frame buffering.
"""

import os
import time
import threading
import logging
from typing import Optional
import cv2
import numpy as np

# Set OpenCV FFmpeg RTSP options to use TCP transport and 5-second socket timeout
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"

logger = logging.getLogger("dahua_render.rtsp")


class RTSPChannelStream:
    """Manages continuous background frame capture from a single RTSP camera stream."""

    def __init__(self, channel: int, rtsp_url: str):
        self.channel = channel
        self.rtsp_url = rtsp_url
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0.0
        self.is_connected = False
        self.reconnect_count = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"RTSP-CH{self.channel}")
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def _capture_loop(self):
        while self.running:
            print(f"[RTSP-CH{self.channel}] Connecting to stream...")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                print(f"[RTSP-CH{self.channel}] Failed to open RTSP stream. Reconnecting in 3s...")
                self.is_connected = False
                time.sleep(3.0)
                continue

            self.is_connected = True
            print(f"[RTSP-CH{self.channel}] Connected successfully!")

            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    print(f"[RTSP-CH{self.channel}] Stream disconnected / dropped frame. Reconnecting...")
                    self.is_connected = False
                    self.reconnect_count += 1
                    break

                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()

                # Yield briefly
                time.sleep(0.01)

            cap.release()
            time.sleep(1.0)
