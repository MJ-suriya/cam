"""
Enterprise Web Application for Dahua 3-Camera AI Face Detection & Attendance.
Streams NVR 2 Channels 30, 31, and 32 with live face recognition overlays and real-time WebSocket check-in alerts.
"""

import os
import json
import time
import shutil
import asyncio
from pathlib import Path
from typing import List
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from config.settings import Config
from config.channels_config import ACTIVE_CHANNELS, get_active_channels, get_channels_info
from core.multi_cam_engine import MultiCameraAttendanceManager
from core.face_engine import (
    EMPLOYEES_JSON, ATTENDANCE_JSON, ATTENDANCE_CSV,
    PROFILE_DB_DIR, SAVED_FACES_DIR
)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Dahua AI Multi-Camera Face Attendance System")

# Mount Static and Asset directories
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")
app.mount("/profiledb", StaticFiles(directory=str(PROFILE_DB_DIR)), name="profiledb")
app.mount("/saved_faces", StaticFiles(directory=str(SAVED_FACES_DIR)), name="saved_faces")

templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

# Initialize Multi-Camera AI Manager
multi_cam_mgr = MultiCameraAttendanceManager()

# Connected WebSocket Clients for Real-Time Ticker
connected_clients: List[WebSocket] = []
main_loop = None


def on_attendance_event(record: dict):
    """Callback fired when an employee is recognized and attendance is marked."""
    global main_loop
    message = json.dumps({"type": "new_attendance", "record": record})
    
    async def _send(ws, msg):
        try:
            await ws.send_text(msg)
        except Exception:
            pass

    if main_loop and main_loop.is_running():
        for ws in list(connected_clients):
            try:
                asyncio.run_coroutine_threadsafe(_send(ws, message), main_loop)
            except Exception:
                pass


multi_cam_mgr.add_attendance_listener(on_attendance_event)


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    print("[Web] Starting Dahua 3-Camera Attendance Engine...")
    multi_cam_mgr.start()


@app.on_event("shutdown")
def shutdown_event():
    print("[Web] Stopping Camera Streams...")
    multi_cam_mgr.stop()


@app.get("/")
async def index():
    return FileResponse(
        BASE_DIR / "web" / "templates" / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


def mjpeg_generator(channel: int):
    """Generator for streaming MJPEG video feed for a camera channel."""
    while True:
        jpeg_bytes = multi_cam_mgr.get_latest_jpeg(channel)
        if jpeg_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        time.sleep(0.04)  # ~25 FPS delivery


@app.get("/api/channels")
async def get_channels():
    """Return all active camera channels and zone labels configured in config/channels_config.py."""
    return get_channels_info()


@app.get("/video_feed/{channel}")
async def video_feed(channel: int):
    if channel not in ACTIVE_CHANNELS:
        return JSONResponse({"error": f"Invalid channel. Allowed: {list(ACTIVE_CHANNELS.keys())}"}, status_code=400)
    return StreamingResponse(
        mjpeg_generator(channel),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/attendance/today")
async def get_today_attendance():
    today_date = time.strftime("%Y-%m-%d")
    records = []
    if ATTENDANCE_JSON.exists():
        try:
            with open(ATTENDANCE_JSON, "r", encoding="utf-8") as f:
                all_records = json.load(f)
                records = [r for r in all_records if r.get("date") == today_date]
        except Exception:
            records = []
    # Return in reverse chronological order (newest first)
    return {"date": today_date, "total": len(records), "records": list(reversed(records))}


@app.get("/api/attendance/export")
async def export_attendance_csv():
    if not ATTENDANCE_CSV.exists():
        return JSONResponse({"error": "No attendance data available yet."}, status_code=404)
    return FileResponse(
        path=str(ATTENDANCE_CSV),
        filename=f"attendance_report_{time.strftime('%Y%m%d')}.csv",
        media_type="text/csv"
    )


@app.get("/api/employees")
async def get_employees():
    employees = []
    if EMPLOYEES_JSON.exists():
        try:
            with open(EMPLOYEES_JSON, "r", encoding="utf-8") as f:
                employees = json.load(f)
        except Exception:
            employees = []
    return employees


@app.post("/api/employees/enroll")
async def enroll_employee(
    ecno: str = Form(...),
    name: str = Form(...),
    photo: UploadFile = File(...)
):
    try:
        ecno = ecno.strip()
        name = name.strip()
        ext = Path(photo.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpeg"

        filename = f"{ecno}{ext}"
        dest_path = PROFILE_DB_DIR / filename

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)

        # Reload face recognition embeddings
        multi_cam_mgr.face_engine.reload_profiles()

        return {"success": True, "message": f"Enrolled {name} ({ecno}) successfully", "image": filename}
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@app.get("/api/stats")
async def get_stats():
    return {
        "channels": [30, 31, 32],
        "port": 37779,
        "detections": multi_cam_mgr.stats["total_detections"],
        "matches": multi_cam_mgr.stats["total_matches"],
        "profiles_count": len(multi_cam_mgr.face_engine.profile_embeddings)
    }


@app.get("/api/verification/today")
async def get_today_verification():
    ledger = multi_cam_mgr.verification_mgr.get_today_ledger()
    active_windows = multi_cam_mgr.verification_mgr.get_active_windows_status()
    today_date = time.strftime("%Y-%m-%d")
    return {
        "date": today_date,
        "total_employees": len(ledger),
        "active_windows_count": len(active_windows),
        "records": ledger,
        "active_windows": active_windows
    }


@app.get("/api/verification/active_windows")
async def get_active_windows():
    return {
        "active_windows": multi_cam_mgr.verification_mgr.get_active_windows_status()
    }


@app.get("/api/config/verification")
async def get_verification_config():
    return multi_cam_mgr.verification_mgr.config.settings


@app.post("/api/config/verification")
async def update_verification_config(request: Request):
    try:
        body = await request.json()
        updated = multi_cam_mgr.verification_mgr.config.update(body)
        return {"success": True, "settings": updated}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@app.get("/api/fingerprint/punches")
async def get_company_punches():
    """Read-only view of the company fingerprint database."""
    punches = multi_cam_mgr.verification_mgr.fp_reader.read_all_punches()
    return {"total": len(punches), "punches": list(reversed(punches))}


@app.post("/api/fingerprint/simulate_punch")
async def simulate_fingerprint_punch(request: Request):
    """Admin tool: Simulates an employee biometric punch on the fingerprint terminal."""
    try:
        body = await request.json()
        ecno = str(body.get("ecno", "")).strip()
        device_id = str(body.get("device_id", "FP_TERMINAL_01")).strip()
        if not ecno:
            return JSONResponse({"success": False, "error": "Employee ID (ecno) is required."}, status_code=400)

        punch = multi_cam_mgr.verification_mgr.fp_reader.simulate_punch_for_testing(ecno, device_id)
        return {"success": True, "punch": punch, "message": f"Biometric punch recorded for ID: {ecno}"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.put("/api/verification/record/{date_str}/{ecno}")
async def update_verification_record(date_str: str, ecno: str, request: Request):
    """Allows altering an existing verification record."""
    try:
        body = await request.json()
        updated = multi_cam_mgr.verification_mgr.update_record(date_str, ecno, body)
        if updated:
            return {"success": True, "record": updated}
        return JSONResponse({"success": False, "error": "Record not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@app.delete("/api/verification/record/{date_str}/{ecno}")
async def delete_verification_record(date_str: str, ecno: str):
    """Allows deleting a verification record."""
    success = multi_cam_mgr.verification_mgr.delete_record(date_str, ecno)
    if success:
        return {"success": True, "message": f"Deleted verification record for {ecno} on {date_str}"}
    return JSONResponse({"success": False, "error": "Record not found"}, status_code=404)


@app.delete("/api/attendance/record/{date_str}/{ecno}")
async def delete_attendance_record(date_str: str, ecno: str):
    """Allows deleting an attendance log record."""
    if ATTENDANCE_JSON.exists():
        try:
            with open(ATTENDANCE_JSON, "r", encoding="utf-8") as f:
                records = json.load(f)
            new_records = [r for r in records if not (r.get("ecno") == ecno and r.get("date") == date_str)]
            with open(ATTENDANCE_JSON, "w", encoding="utf-8") as f:
                json.dump(new_records, f, indent=2)
            return {"success": True, "message": f"Deleted attendance record for {ecno}"}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    return JSONResponse({"success": False, "error": "File not found"}, status_code=404)


@app.get("/api/verification/export")
async def export_verification_csv():
    from core.verification_engine import AI_LEDGER_CSV
    if not AI_LEDGER_CSV.exists():
        return JSONResponse({"error": "No verification ledger data available yet."}, status_code=404)
    return FileResponse(
        path=str(AI_LEDGER_CSV),
        filename=f"attendance_verification_report_{time.strftime('%Y%m%d')}.csv",
        media_type="text/csv"
    )


@app.websocket("/ws/attendance")
async def websocket_attendance(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Keep-alive heartbeat
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
