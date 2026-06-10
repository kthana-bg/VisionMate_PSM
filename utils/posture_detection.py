"""
Posture detection module.
Uses body landmark coordinates to compute head-neck-shoulder angles
and runs inference using whichever of the three posture models is selected.

Model mapping (per spec):
  A2 — Angle-Based (rule-based)    → classify_posture_by_angle
  B2 — BlazePose / MoveNet DNN     → run_posture_model_inference
  C2 — Custom LSTM                 → run_posture_model_inference

Rule-based angle classification is ONLY used when model_name == "A2 - Angle-Based".
No predefined angle threshold runs in the background for B2 or C2.
"""

import time
import numpy as np
import cv2
import math

# ── Angle constant — used ONLY when A2 (Angle-Based) is the selected model ──
_A2_SLOUCH_ANGLE_THRESHOLD = 20.0   # degrees above vertical → slouching

# MediaPipe Pose landmark indices used for posture calculation
NOSE_IDX           = 0
LEFT_EAR_IDX       = 7
RIGHT_EAR_IDX      = 8
LEFT_SHOULDER_IDX  = 11
RIGHT_SHOULDER_IDX = 12


def calculate_angle(point_a: tuple, point_b: tuple, point_c: tuple) -> float:
    """
    Calculate the angle at point_b formed by the line segments b->a and b->c.
    All points are (x, y) pixel coordinates.
    Returns angle in degrees.
    """
    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cosine_angle))
    return angle


def calculate_neck_tilt_angle(
    ear_midpoint: tuple,
    shoulder_midpoint: tuple
) -> float:
    """
    Calculate the neck tilt angle relative to vertical.
    ear_midpoint and shoulder_midpoint are (x, y) pixel coords.
    A small angle means the head is upright; a large angle means forward lean.
    Always computed for display purposes regardless of selected model.
    """
    dx = ear_midpoint[0] - shoulder_midpoint[0]
    dy = shoulder_midpoint[1] - ear_midpoint[1]  # positive = upward

    angle = math.degrees(math.atan2(abs(dx), max(dy, 1)))
    return angle


def extract_pose_landmarks(pose_results, image_width: int, image_height: int) -> dict | None:
    """
    Extract the key landmark pixel positions from a MediaPipe Pose result.
    Returns a dict of named points, or None if landmarks are not detected.
    """
    if not pose_results.pose_landmarks:
        return None

    lm = pose_results.pose_landmarks.landmark

    def to_pixel(idx):
        return (int(lm[idx].x * image_width), int(lm[idx].y * image_height))

    return {
        "nose":           to_pixel(NOSE_IDX),
        "left_ear":       to_pixel(LEFT_EAR_IDX),
        "right_ear":      to_pixel(RIGHT_EAR_IDX),
        "left_shoulder":  to_pixel(LEFT_SHOULDER_IDX),
        "right_shoulder": to_pixel(RIGHT_SHOULDER_IDX),
    }


def classify_posture_by_angle(angle: float) -> str:
    """
    A2 ONLY: Rule-based posture classification from neck tilt angle.
    Called ONLY when the user has explicitly selected A2 - Angle-Based.
    Returns 'Good' or 'Slouching'.
    """
    if angle > _A2_SLOUCH_ANGLE_THRESHOLD:
        return "Slouching"
    return "Good"


def extract_landmark_feature_vector(landmarks: dict) -> np.ndarray:
    """
    Build a flat feature vector from landmark coordinates.
    Used as input for LSTM/DNN posture models (B2, C2).
    Normalizes all coordinates relative to shoulder midpoint so the
    feature is scale-invariant.
    """
    shoulder_mid_x = (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) / 2.0
    shoulder_mid_y = (landmarks["left_shoulder"][1] + landmarks["right_shoulder"][1]) / 2.0

    shoulder_width = max(
        abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]),
        1
    )

    points = [
        landmarks["nose"],
        landmarks["left_ear"],
        landmarks["right_ear"],
        landmarks["left_shoulder"],
        landmarks["right_shoulder"],
    ]

    features = []
    for (x, y) in points:
        features.append((x - shoulder_mid_x) / shoulder_width)
        features.append((y - shoulder_mid_y) / shoulder_width)

    return np.array(features, dtype=np.float32)


def run_posture_model_inference(model, feature_vector: np.ndarray, model_name: str) -> dict:
    """
    Run posture inference using the selected ML model (B2 or C2).
    This must ONLY be called for B2/C2 — never falls back to angle rules here.
    Returns:
        label      - 'Good' or 'Slouching'
        confidence - float in [0, 1]
        latency_ms - inference time
    """
    start = time.perf_counter()

    if "B2" in model_name or "BlazePose" in model_name or "MoveNet" in model_name or "YOLO" in model_name:
        input_tensor = feature_vector.reshape(1, -1)
        prediction = model.predict(input_tensor, verbose=0)[0]
        slouch_conf = float(prediction[1]) if len(prediction) > 1 else float(prediction[0])
        label = "Slouching" if slouch_conf > 0.5 else "Good"
        confidence = slouch_conf if label == "Slouching" else 1.0 - slouch_conf
    else:
        # Custom LSTM / DNN (C2)
        if len(model.input_shape) == 3:
            # LSTM expects (batch, timesteps, features)
            input_tensor = feature_vector.reshape(1, 1, -1)
        else:
            input_tensor = feature_vector.reshape(1, -1)
        prediction = model.predict(input_tensor, verbose=0)[0]
        slouch_conf = float(prediction[1]) if len(prediction) > 1 else float(prediction[0])
        label = "Slouching" if slouch_conf > 0.5 else "Good"
        confidence = slouch_conf if label == "Slouching" else 1.0 - slouch_conf

    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "label": label,
        "confidence": confidence,
        "latency_ms": latency_ms
    }


def draw_posture_overlay(
    frame: np.ndarray,
    landmarks: dict,
    angle: float,
    status: str
):
    """
    Draw posture skeleton lines and angle text on the frame in-place.
    Green lines = good posture, Red lines = slouching.
    """
    color = (0, 255, 0) if status == "Good" else (0, 0, 255)

    nose           = landmarks["nose"]
    left_ear       = landmarks["left_ear"]
    right_ear      = landmarks["right_ear"]
    left_shoulder  = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]

    ear_mid = (
        (left_ear[0] + right_ear[0]) // 2,
        (left_ear[1] + right_ear[1]) // 2
    )
    shoulder_mid = (
        (left_shoulder[0] + right_shoulder[0]) // 2,
        (left_shoulder[1] + right_shoulder[1]) // 2
    )

    cv2.line(frame, ear_mid, shoulder_mid, color, 2)
    cv2.line(frame, left_shoulder, right_shoulder, color, 2)
    cv2.circle(frame, nose, 5, color, -1)
    cv2.circle(frame, ear_mid, 5, color, -1)
    cv2.circle(frame, shoulder_mid, 5, color, -1)

    cv2.putText(
        frame,
        f"Angle: {angle:.1f}deg",
        (shoulder_mid[0] - 60, shoulder_mid[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1
    )
