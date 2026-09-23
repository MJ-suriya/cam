"""
Channel configuration for Dahua AI Face & Fingerprint Attendance System on Render.
"""

from typing import Dict, List

# Active Channels configuration
ACTIVE_CHANNELS: Dict[int, str] = {
    30: "Turnstile (Main Entrance)",
    31: "Reception Lobby (Central)",
    32: "Access Corridor (Ground Floor)",
    33: "Gate / Entrance Channel 33",
    34: "Channel 34 Surveillance",
    53: "Channel 53 Surveillance"
}


def get_active_channels() -> List[int]:
    """Returns list of active channel IDs."""
    return list(ACTIVE_CHANNELS.keys())


def get_channel_name(channel_id: int) -> str:
    """Returns zone name for a specific channel ID."""
    return ACTIVE_CHANNELS.get(channel_id, f"Channel {channel_id}")


def get_channels_info() -> List[Dict]:
    """Returns structured list of channel information for web API."""
    return [{"id": ch, "name": name} for ch, name in ACTIVE_CHANNELS.items()]
