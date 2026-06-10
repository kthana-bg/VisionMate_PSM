"""
Eye strain detection module.
Calculates Eye Aspect Ratio (EAR) from MediaPipe face mesh landmarks
and runs inference using whichever of the three eye models is selected.

Model mapping (per spec):
  A1 — EAR Threshold (rule-based)      → classify_eye_status_by_ear
  B1 — MobileNetV2                     → run_eye_model_inference
  C1 — Custom CNN                      → run_eye_model_inference

Rule-based EAR classification is ONLY used when model_name == "A1 - EAR Threshold".
No predefined EAR threshold runs in the background for B1 or C1.
"""

import time
import numpy as np
import cv2
from scipy.spatial import distance as dist

# ── EAR constants — used ONLY when A1 (EAR Threshold) is the selected model ──
_A1_EAR_THRESHOLD    = 0.18   # below this → closed/drowsy
_A1_EAR_CONSEC_FRAMES = 30    # consecutive frames before flagging Strained

# MediaPipe face mesh landmark indices for left and right eyes
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio for one eye.
    eye_landmarks: array of shape (6, 2) with (x, y) pixel coords.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Always computed for display purposes regardless of selected model.
    """
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def extract_eye_landmarks(face_landmarks, image_width: int, image_height: int):
    """
    Extract left and right eye landmark pixel coordinates from MediaPipe results.
    Returns two arrays of shape (6, 2).
    """
    def get_coords(indices):
        points = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            x  = int(lm.x * image_width)
            y  = int(lm.y * image_height)
            points.append((x, y))
        return np.array(points, dtype=np.float64)

    return get_coords(LEFT_EYE_INDICES), get_coords(RIGHT_EYE_INDICES)


def get_eye_roi(frame: np.ndarray, eye_landmarks: np.ndarray, padding: int = 10):
    """
    Crop the region of interest around an eye for CNN inference.
    Returns a resized 64x32 BGR image, or None if crop fails.
    """
    x_coords = eye_landmarks[:, 0].astype(int)
    y_coords = eye_landmarks[:, 1].astype(int)
    x_min = max(0, x_coords.min() - padding)
    x_max = min(frame.shape[1], x_coords.max() + padding)
    y_min = max(0, y_coords.min() - padding)
    y_max = min(frame.shape[0], y_coords.max() + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (64, 32))


def preprocess_eye_image(roi: np.ndarray) -> np.ndarray:
    """
    Normalize and expand dimensions for model input.
    Input:  (32, 64, 3) uint8
    Output: (1, 32, 64, 3) float32 in [0, 1]
    """
    img = roi.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def run_eye_model_inference(model, roi: np.ndarray, model_name: str) -> dict:
    """
    Run inference on a single eye ROI using the selected ML model (B1 or C1).
    This must ONLY be called for B1/C1 — never falls back to EAR rules here.
    Returns label, confidence, and latency_ms.
    """
    start = time.perf_counter()

    input_tensor = preprocess_eye_image(roi)
    prediction   = model.predict(input_tensor, verbose=0)[0]

    if len(prediction) == 1:
        confidence = float(prediction[0])
        label      = "Strained" if confidence > 0.5 else "Normal"
        if label == "Normal":
            confidence = 1.0 - confidence
    else:
        strained_conf = float(prediction[1])
        label         = "Strained" if strained_conf > 0.5 else "Normal"
        confidence    = strained_conf if label == "Strained" else float(prediction[0])

    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"label": label, "confidence": confidence, "latency_ms": latency_ms}


def classify_eye_status_by_ear(ear: float, consec_counter: int) -> tuple:
    """
    A1 ONLY: Rule-based classification using EAR value.
    Called ONLY when the user has explicitly selected A1 - EAR Threshold.
    Returns (status_string, updated_consec_counter).
    """
    if ear < _A1_EAR_THRESHOLD:
        consec_counter += 1
    else:
        consec_counter = 0   # reset immediately when eye opens

    if consec_counter >= _A1_EAR_CONSEC_FRAMES:
        status = "Strained"
    elif ear < _A1_EAR_THRESHOLD:
        status = "Blinking"
    else:
        status = "Normal"

    return status, consec_counter


def draw_eye_landmarks(frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray):
    """Draw convex hull outlines around both eyes on the frame in-place."""
    for eye_pts in [left_eye, right_eye]:
        hull = cv2.convexHull(eye_pts.astype(np.int32))
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
