"""
Configuration Module for Dahua NVR Camera Monitor (Render / Linux Docker).
Loads settings from environment variables with safe defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
if (BASE_DIR / ".env").exists():
    load_dotenv(dotenv_path=BASE_DIR / ".env")


class Config:
    """NVR and Application Configuration for Cloud / Render."""

    # Network Settings
    NVR_IP: str = os.getenv("NVR_IP", "").strip()
    RTSP_PORT: int = int(os.getenv("RTSP_PORT", "554"))
    HTTP_PORT: int = int(os.getenv("HTTP_PORT", "80"))
    WEB_PORT: int = int(os.getenv("PORT", "8000"))

    # Authentication
    NVR_USERNAME: str = os.getenv("NVR_USERNAME", "admin").strip()
    NVR_PASSWORD: str = os.getenv("NVR_PASSWORD", "").strip()

    # Stream Settings (0=Main high-res, 1=Sub stream efficient for multi-cam AI)
    DEFAULT_SUBTYPE: int = int(os.getenv("DEFAULT_SUBTYPE", "1"))

    # Timeouts
    RTSP_TIMEOUT: float = float(os.getenv("RTSP_TIMEOUT", "8.0"))

    @classmethod
    def get_rtsp_url(cls, channel: int, subtype: int = 1) -> str:
        """Constructs secure Dahua RTSP URL for a given channel and stream subtype."""
        user = cls.NVR_USERNAME
        pwd = cls.NVR_PASSWORD
        ip = cls.NVR_IP
        port = cls.RTSP_PORT
        return f"rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
