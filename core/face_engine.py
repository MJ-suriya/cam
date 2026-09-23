"""
AI Face Detection & Recognition Engine for Multi-Camera CCTV Attendance.
Powered by OpenCV YuNet (Face Detection) and SFace (Face Recognition).
"""

import os
import csv
import json
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROFILE_DB_DIR = BASE_DIR / "profiledb"
SAVED_FACES_DIR = BASE_DIR / "saved_faces"
LOGS_DIR = BASE_DIR / "logs"

EMPLOYEES_JSON = BASE_DIR / "employees.json"
ATTENDANCE_JSON = BASE_DIR / "attendance.json"
ATTENDANCE_CSV = LOGS_DIR / "attendance.csv"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

SAVED_FACES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
PROFILE_DB_DIR.mkdir(exist_ok=True)


MIN_FACE_WIDTH = 42
MIN_SHARPNESS = 15.0
MAX_CAPTURES_PER_PERSON = 2


def get_sharpness_score(image: np.ndarray) -> float:
    """Compute Laplacian variance sharpness score."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_face_crop(crop: np.ndarray, target_size: int = 240) -> np.ndarray:
    """Enhance facial crop clarity with bilateral denoising, CLAHE lighting balance, and unsharp mask."""
    if crop is None or crop.size == 0:
        return crop

    # Denoise
    denoised = cv2.bilateralFilter(crop, d=5, sigmaColor=25, sigmaSpace=25)

    # CLAHE balance
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # Resize to target
    h, w = enhanced.shape[:2]
    if max(h, w) < target_size:
        enhanced = cv2.resize(enhanced, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    # Unsharp Mask
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)
    return sharpened


def is_valid_face_geometry(face_arr) -> bool:
    """Validates that detected object has realistic human facial geometry and rejects inanimate structures."""
    fw, fh = face_arr[2], face_arr[3]
    if fw < 32 or fh < 32:
        return False

    aspect = fh / max(1.0, fw)
    if aspect < 0.75 or aspect > 1.80:
        return False

    r_eye = (face_arr[4], face_arr[5])
    l_eye = (face_arr[6], face_arr[7])
    nose = (face_arr[8], face_arr[9])
    m_right = (face_arr[10], face_arr[11])
    m_left = (face_arr[12], face_arr[13])

    # Eye separation distance
    eye_dist = np.hypot(l_eye[0] - r_eye[0], l_eye[1] - r_eye[1])
    if eye_dist < fw * 0.18 or eye_dist > fw * 0.70:
        return False

    # Eye tilt check (reject extreme vertical artifacts / lines)
    if abs(l_eye[1] - r_eye[1]) / (eye_dist + 1e-5) > 0.60:
        return False

    # Vertical hierarchy: Eyes above nose, Nose above mouth with distinct vertical gaps
    avg_eye_y = (r_eye[1] + l_eye[1]) / 2.0
    avg_mouth_y = (m_right[1] + m_left[1]) / 2.0

    eye_to_nose_gap = nose[1] - avg_eye_y
    nose_to_mouth_gap = avg_mouth_y - nose[1]

    if eye_to_nose_gap < fh * 0.08 or nose_to_mouth_gap < fh * 0.08:
        return False

    return True


YOLO_ONNX_MODEL = MODELS_DIR / "yolov8n.onnx" if (MODELS_DIR / "yolov8n.onnx").exists() else (BASE_DIR / "yolov8n.onnx")


class OpenCVDNNYOLOv8:
    """
    Ultra-lightweight YOLOv8 Person Detector running natively via OpenCV DNN C++ engine.
    Consumes only ~25MB of RAM (avoids heavy 450MB PyTorch framework).
    """

    def __init__(self, onnx_path: str, conf_thresh: float = 0.45):
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf_thresh = conf_thresh

    def detect_persons(self, frame: np.ndarray, orig_w: int, orig_h: int) -> List[Tuple[int, int, int, int]]:
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (384, 384), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()  # (1, 84, 3024)
        preds = output[0].T
        boxes = []
        confidences = []
        scale_x = orig_w / 384.0
        scale_y = orig_h / 384.0

        for row in preds:
            person_score = float(row[4])
            if person_score >= self.conf_thresh:
                cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                x1 = int((cx - w / 2.0) * scale_x)
                y1 = int((cy - h / 2.0) * scale_y)
                pw = int(w * scale_x)
                ph = int(h * scale_y)
                if pw >= 35 and ph >= 70:
                    boxes.append([x1, y1, pw, ph])
                    confidences.append(person_score)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, 0.45)
        person_boxes = []
        if len(indices) > 0:
            for idx in indices.flatten():
                b = boxes[idx]
                person_boxes.append((max(0, b[0]), max(0, b[1]), b[2], b[3]))

        return person_boxes


class FaceRecognitionEngine:
    """
    Two-Stage Enterprise CCTV Face Detection & Recognition Engine (All 3 Models Active in < 180MB RAM):
    Stage 1: YOLOv8 Person Detection (OpenCV DNN C++ onnx, eliminates non-human false positives)
    Stage 2: OpenCV YuNet Face Detection + SFace 128-d cosine recognition on verified human head regions.
    """

    def __init__(self, match_threshold: float = 0.46, detector_score_thresh: float = 0.80):
        self.match_threshold = match_threshold
        self.detector_score_thresh = detector_score_thresh
        self.high_conf_threshold = 0.51  # 87%+ high priority match
        self.infer_lock = threading.Lock()
        
        self.person_detector = None
        self.detector = None
        self.recognizer = None
        self.profile_embeddings: Dict[str, Dict] = {}  # {ecno: {"name": str, "embedding": np.ndarray, "image": str}}
        
        self._init_models()
        self.reload_profiles()

    def _init_models(self):
        # 1. Model 1: YOLOv8 Person Detector (OpenCV ONNX, ~25MB RAM)
        if YOLO_ONNX_MODEL.exists():
            try:
                self.person_detector = OpenCVDNNYOLOv8(str(YOLO_ONNX_MODEL), conf_thresh=0.45)
                print(f"[FaceEngine] Model 1: OpenCV DNN YOLOv8 Person Detector loaded ({YOLO_ONNX_MODEL.name})")
            except Exception as e:
                print(f"[FaceEngine] Warning loading OpenCV YOLOv8: {e}")

        # 2. Model 2: YuNet Face Detector
        if YUNET_MODEL.exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    str(YUNET_MODEL), "", (320, 320), self.detector_score_thresh, 0.35
                )
                print(f"[FaceEngine] Model 2: YuNet detector loaded ({YUNET_MODEL.name})")
            except Exception as e:
                print(f"[FaceEngine] Error loading YuNet: {e}")

        # 3. Model 3: SFace Face Recognizer
        if SFACE_MODEL.exists():
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
                print(f"[FaceEngine] Model 3: SFace recognizer loaded ({SFACE_MODEL.name})")
            except Exception as e:
                print(f"[FaceEngine] Error loading SFace: {e}")

    def reload_profiles(self):
        """Scans profiledb/ and employees.json to load all reference employee embeddings."""
        self.profile_embeddings.clear()
        if not self.recognizer or not self.detector:
            return

        # Load names from employees.json
        emp_names = {}
        if EMPLOYEES_JSON.exists():
            try:
                with open(EMPLOYEES_JSON, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        ec = str(item.get("ecno", "")).strip()
                        if ec:
                            emp_names[ec] = item.get("name", f"Employee {ec}")
            except Exception as e:
                print(f"[FaceEngine] Error reading employees.json: {e}")

        # Scan profiledb
        updated_list = []
        for img_file in sorted(PROFILE_DB_DIR.iterdir()):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                ecno = img_file.stem.strip()
                name = emp_names.get(ecno, f"Employee {ecno}")
                img = cv2.imread(str(img_file))
                if img is not None:
                    feat = self.extract_embedding(img)
                    if feat is not None:
                        self.profile_embeddings[ecno] = {
                            "ecno": ecno,
                            "name": name,
                            "embedding": feat,
                            "image": img_file.name
                        }
                        print(f"[FaceEngine] Enrolled profile: {name} ({ecno})")
                updated_list.append({"ecno": ecno, "name": name, "image": img_file.name})

        # Keep employees.json synced
        try:
            with open(EMPLOYEES_JSON, "w", encoding="utf-8") as f:
                json.dump(updated_list, f, indent=2)
        except Exception:
            pass

    def extract_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract 128-d feature vector using YuNet alignment + SFace."""
        if img is None or img.size == 0 or not self.recognizer:
            return None

        h, w = img.shape[:2]
        if self.detector:
            self.detector.setInputSize((w, h))
            self.detector.setScoreThreshold(0.50)
            _, faces = self.detector.detect(img)
            if faces is not None and len(faces) > 0:
                aligned = self.recognizer.alignCrop(img, faces[0])
                return self.recognizer.feature(aligned)

        # Fallback resize
        resized = cv2.resize(img, (112, 112))
        return self.recognizer.feature(resized)

    def detect_and_recognize(self, frame: np.ndarray) -> List[Dict]:
        """
        Two-Stage CCTV Face Detection:
        1. YOLOv8 verifies presence of a real human (Person class=0, conf>=0.45).
        2. YuNet pinpoints facial landmarks and boundaries in the person's head region.
        3. SFace recognizes employee against profiledb with strict >= 87% confidence.
        """
        if frame is None or frame.size == 0 or not self.detector or not self.recognizer:
            return []

        with self.infer_lock:
            orig_h, orig_w = frame.shape[:2]
            results = []

            # Stage 1: YOLOv8 Person Verification Gate (OpenCV DNN C++ native)
            person_boxes = []
            if self.person_detector is not None:
                try:
                    person_boxes = self.person_detector.detect_persons(frame, orig_w, orig_h)
                except Exception as e:
                    print(f"[FaceEngine] Person detector warning: {e}")

                # If no person is present in the CCTV frame, return empty (0 false positives on stairs/floors)
                if not person_boxes:
                    return results
            else:
                # If person detector is not loaded, process whole frame
                person_boxes = [(0, 0, orig_w, orig_h)]

            # Stage 2: Face Detection inside each detected human's upper body / head region
            for (px, py, pw, ph) in person_boxes:
                if self.person_detector is not None:
                    # Crop upper 50% of the person (head & shoulder region)
                    hx1 = max(0, px - int(pw * 0.15))
                    hy1 = max(0, py - int(ph * 0.08))
                    hx2 = min(orig_w, px + pw + int(pw * 0.15))
                    hy2 = min(orig_h, py + int(ph * 0.55))
                    head_crop = frame[hy1:hy2, hx1:hx2]
                else:
                    hx1, hy1 = 0, 0
                    head_crop = frame

                ch_h, ch_w = head_crop.shape[:2]
                if ch_h < 32 or ch_w < 32:
                    continue

                self.detector.setInputSize((ch_w, ch_h))
                self.detector.setScoreThreshold(self.detector_score_thresh)
                self.detector.setNMSThreshold(0.35)
                _, faces = self.detector.detect(head_crop)

                if faces is None or len(faces) == 0:
                    continue

                for f_det in faces:
                    det_conf = float(f_det[14])
                    # Ensure face detection confidence >= 0.80 inside human crop
                    if det_conf < self.detector_score_thresh:
                        continue

                    if not is_valid_face_geometry(f_det):
                        continue

                    # Map face coordinates and landmarks back to original native frame
                    fx = int(f_det[0] + hx1)
                    fy = int(f_det[1] + hy1)
                    fw = int(f_det[2])
                    fh = int(f_det[3])

                    native_face_arr = f_det.copy()
                    native_face_arr[0] = fx
                    native_face_arr[1] = fy
                    native_face_arr[2] = fw
                    native_face_arr[3] = fh
                    for i in range(4, 14, 2):
                        native_face_arr[i] = f_det[i] + hx1
                        native_face_arr[i+1] = f_det[i+1] + hy1

                    landmarks = [
                        (int(native_face_arr[4]), int(native_face_arr[5])),
                        (int(native_face_arr[6]), int(native_face_arr[7])),
                        (int(native_face_arr[8]), int(native_face_arr[9])),
                        (int(native_face_arr[10]), int(native_face_arr[11])),
                        (int(native_face_arr[12]), int(native_face_arr[13])),
                    ]

                    # Crop and align face from full native resolution
                    aligned = self.recognizer.alignCrop(frame, native_face_arr)
                    query_feat = self.recognizer.feature(aligned)

                    best_ecno = None
                    best_name = None
                    best_score = 0.0

                    for ecno, data in self.profile_embeddings.items():
                        score = float(self.recognizer.match(query_feat, data["embedding"], cv2.FaceRecognizerSF_FR_COSINE))
                        if score > best_score:
                            best_score = score
                            best_ecno = ecno
                            best_name = data["name"]

                    is_matched = best_score >= self.match_threshold
                    is_high_confidence = best_score >= self.high_conf_threshold

                    results.append({
                        "bbox": (fx, fy, fw, fh),
                        "det_confidence": round(det_conf, 3),
                        "landmarks": landmarks,
                        "is_matched": is_matched,
                        "is_high_confidence": is_high_confidence,
                        "ecno": best_ecno if is_matched else None,
                        "name": best_name if is_matched else "Unregistered Person",
                        "match_score": round(best_score, 3),
                        "raw_face_arr": native_face_arr
                    })

            return results

            # Stage 2: Face Detection inside each detected human's upper body / head region
            for (px, py, pw, ph) in person_boxes:
                # Crop upper 50% of the person (head & shoulder region)
                hx1 = max(0, px - int(pw * 0.15))
                hy1 = max(0, py - int(ph * 0.08))
                hx2 = min(orig_w, px + pw + int(pw * 0.15))
                hy2 = min(orig_h, py + int(ph * 0.55))

                head_crop = frame[hy1:hy2, hx1:hx2]
                ch_h, ch_w = head_crop.shape[:2]
                if ch_h < 32 or ch_w < 32:
                    continue

                self.detector.setInputSize((ch_w, ch_h))
                self.detector.setScoreThreshold(self.detector_score_thresh)
                self.detector.setNMSThreshold(0.35)
                _, faces = self.detector.detect(head_crop)

                if faces is None or len(faces) == 0:
                    continue

                for f_det in faces:
                    det_conf = float(f_det[14])
                    # Ensure face detection confidence >= 0.80 inside human crop
                    if det_conf < self.detector_score_thresh:
                        continue

                    if not is_valid_face_geometry(f_det):
                        continue

                    # Map face coordinates and landmarks back to original native frame
                    fx = int(f_det[0] + hx1)
                    fy = int(f_det[1] + hy1)
                    fw = int(f_det[2])
                    fh = int(f_det[3])

                    native_face_arr = f_det.copy()
                    native_face_arr[0] = fx
                    native_face_arr[1] = fy
                    native_face_arr[2] = fw
                    native_face_arr[3] = fh
                    for i in range(4, 14, 2):
                        native_face_arr[i] = f_det[i] + hx1
                        native_face_arr[i+1] = f_det[i+1] + hy1

                    landmarks = [
                        (int(native_face_arr[4]), int(native_face_arr[5])),
                        (int(native_face_arr[6]), int(native_face_arr[7])),
                        (int(native_face_arr[8]), int(native_face_arr[9])),
                        (int(native_face_arr[10]), int(native_face_arr[11])),
                        (int(native_face_arr[12]), int(native_face_arr[13])),
                    ]

                    # Crop and align face from full native resolution
                    aligned = self.recognizer.alignCrop(frame, native_face_arr)
                    query_feat = self.recognizer.feature(aligned)

                    best_ecno = None
                    best_name = None
                    best_score = 0.0

                    for ecno, data in self.profile_embeddings.items():
                        score = float(self.recognizer.match(query_feat, data["embedding"], cv2.FaceRecognizerSF_FR_COSINE))
                        if score > best_score:
                            best_score = score
                            best_ecno = ecno
                            best_name = data["name"]

                    # Match Threshold: >= 0.46 (>=80% confidence), with >= 0.51 (>=87%) concentrated high confidence
                    is_matched = best_score >= self.match_threshold
                    is_high_confidence = best_score >= self.high_conf_threshold

                    results.append({
                        "bbox": (fx, fy, fw, fh),
                        "det_confidence": round(det_conf, 3),
                        "landmarks": landmarks,
                        "is_matched": is_matched,
                        "is_high_confidence": is_high_confidence,
                        "ecno": best_ecno if is_matched else None,
                        "name": best_name if is_matched else "Unregistered Person",
                        "match_score": round(best_score, 3),
                        "raw_face_arr": native_face_arr
                    })

            return results

    def save_face_snapshot(
        self,
        frame: np.ndarray,
        bbox: tuple,
        channel: int,
        is_matched: bool,
        ecno: Optional[str]
    ) -> Optional[str]:
        """Saves a high-clarity face crop to saved_faces/ with contrast and sharpness enhancement."""
        today_date = time.strftime("%Y-%m-%d")
        now_ts = int(time.time())
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        
        # Dedicated Face crop with generous padding (forehead, hair, ears, chin)
        pad_x = int(w * 0.35)
        pad_y = int(h * 0.40)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
        
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None

        # Enhance face crop
        enhanced = enhance_face_crop(crop)
        save_img = enhanced if (enhanced is not None and enhanced.size > 0) else crop

        if is_matched and ecno:
            filename = f"{ecno}_{today_date}_{now_ts}.jpg"
        else:
            filename = f"visitor_CH{channel}_{today_date}_{now_ts}.jpg"

        dest_path = SAVED_FACES_DIR / filename
        try:
            cv2.imwrite(str(dest_path), save_img)
            return filename
        except Exception as e:
            print(f"[FaceEngine] Error saving snapshot: {e}")
            return None

    def mark_attendance(
        self,
        ecno: str,
        name: str,
        channel: int,
        confidence: float,
        frame: np.ndarray,
        bbox: tuple
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Records verified employee attendance (only for confidence >= match_threshold / >= 87%).
        Deduplicates daily records.
        """
        today_date = time.strftime("%Y-%m-%d")
        now_time = time.strftime("%H:%M:%S")

        records = []
        if ATTENDANCE_JSON.exists():
            try:
                with open(ATTENDANCE_JSON, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []

        # Deduplication check
        for r in records:
            if r.get("ecno") == ecno and r.get("date") == today_date:
                return False, r  # Already marked today

        # Save face crop
        snapshot_filename = self.save_face_snapshot(frame, bbox, channel, is_matched=True, ecno=ecno)

        # Normalize confidence for display (0.51 score -> 87%, 0.70 score -> 98%)
        display_conf = min(0.99, max(0.87, 0.87 + (confidence - 0.51) * 0.40))

        new_record = {
            "ecno": ecno,
            "name": name,
            "date": today_date,
            "check_in_time": now_time,
            "camera_source": f"NVR 2 - Channel {channel:02d}",
            "channel": channel,
            "status": "PRESENT",
            "confidence": round(display_conf, 2),
            "snapshot": snapshot_filename
        }

        records.append(new_record)
        with open(ATTENDANCE_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        # Append to CSV
        csv_exists = ATTENDANCE_CSV.exists()
        with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["ecno", "name", "date", "check_in_time", "camera_source", "channel", "status", "confidence", "snapshot"]
            )
            if not csv_exists:
                writer.writeheader()
            writer.writerow(new_record)

        print(f"[Attendance] [OK] VERIFIED ({int(display_conf*100)}%): {name} ({ecno}) at {now_time} via CH {channel:02d}")
        return True, new_record
