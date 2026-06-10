"""
inference_server.py
-------------------
FastAPI sidecar that runs inside the same Render dyno as Streamlit.
The browser POSTs raw JPEG frames here; we run model inference
server-side and return an annotated JPEG + JSON metrics.

Key principle: ONLY the user-selected models are loaded and run.
No background EAR thresholds or angle rules execute unless the user
explicitly chose A1 or A2.
"""

import os, sys, time, base64, threading
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.frame_processor import (
    FrameResult,
    process_frame,
    load_mediapipe_landmarkers,
    compute_health_score,
    EYE_MODEL_C1,
    POSTURE_MODEL_C2,
)
from utils.model_loader import (
    load_selected_eye_model,
    load_selected_posture_model,
)

# ──────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────

class _InferenceState:
    def __init__(self):
        self._lock               = threading.Lock()
        self.eye_model_name      = EYE_MODEL_C1
        self.posture_model_name  = POSTURE_MODEL_C2
        self.eye_model           = None
        self.posture_model       = None
        self.face_landmarker     = None
        self.pose_landmarker     = None
        self._ear_consec         = 0
        self._last_result        = FrameResult()
        self._ready              = False

    def ensure_landmarkers(self):
        with self._lock:
            if self.face_landmarker is None:
                self.face_landmarker, self.pose_landmarker = load_mediapipe_landmarkers()

    def load_initial_models(self):
        """Load only the default-selected models on startup."""
        with self._lock:
            self.eye_model, _     = load_selected_eye_model(self.eye_model_name)
            self.posture_model, _ = load_selected_posture_model(self.posture_model_name)
            self._ready = True
            print(f"[inference_server] Initial models loaded: "
                  f"{self.eye_model_name} + {self.posture_model_name}")

    def set_active_models(self, eye_name: str, posture_name: str):
        """
        Switch to a new model pair.
        Only the newly selected models are loaded; previously loaded
        models are released so they don't consume memory.
        """
        with self._lock:
            if eye_name == self.eye_model_name and posture_name == self.posture_model_name:
                return  # no change

            print(f"[inference_server] Switching to: {eye_name} + {posture_name}")

            # Release old models
            self.eye_model     = None
            self.posture_model = None

            # Load only the new selections
            self.eye_model_name     = eye_name
            self.posture_model_name = posture_name
            self.eye_model, _       = load_selected_eye_model(eye_name)
            self.posture_model, _   = load_selected_posture_model(posture_name)

            # Reset EAR consecutive counter on model switch
            self._ear_consec = 0
            print(f"[inference_server] Models ready.")

    def process(self, frame_bgr: np.ndarray):
        with self._lock:
            result, self._ear_consec = process_frame(
                frame_bgr,
                self.face_landmarker,
                self.pose_landmarker,
                self.eye_model,
                self.eye_model_name,
                self.posture_model,
                self.posture_model_name,
                self._ear_consec,
            )
            self._last_result = result
        return result

    def get_last_result(self) -> FrameResult:
        with self._lock:
            return self._last_result


# Singleton — imported by app.py / monitoring_tab to set models
inference_state = _InferenceState()


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(title="VisionMate Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class FrameRequest(BaseModel):
    image: str        # base64-encoded JPEG
    user_id: str = ""


@app.on_event("startup")
async def _startup():
    """Load MediaPipe landmarkers + default-selected models on startup."""
    print("[inference_server] Loading MediaPipe landmarkers...")
    inference_state.ensure_landmarkers()
    print("[inference_server] Loading selected models...")
    inference_state.load_initial_models()
    print("[inference_server] Ready.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/frame")
def process_incoming_frame(req: FrameRequest):
    """
    Accepts a base64 JPEG frame from the browser.
    Returns annotated JPEG (base64) + metric JSON.
    Results are driven solely by the user-selected models.
    """
    try:
        raw = base64.b64decode(req.image)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image: {e}")

    result = inference_state.process(frame_bgr)

    _, buf = cv2.imencode(".jpg", result.frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    annotated_b64 = base64.b64encode(buf.tobytes()).decode()

    return JSONResponse({
        "annotated_frame":   annotated_b64,
        "eye_status":        result.eye_status,
        "ear_value":         round(result.ear_value, 4),
        "posture_status":    result.posture_status,
        "posture_angle":     round(result.posture_angle, 2),
        "health_score":      round(result.health_score, 1),
        "face_detected":     result.face_detected,
        "eye_latency_ms":    round(result.eye_latency_ms, 2),
        "posture_latency_ms": round(result.posture_latency_ms, 2),
        "eye_model":         inference_state.eye_model_name,
        "posture_model":     inference_state.posture_model_name,
    })


@app.post("/set_models")
def set_models(eye_model: str, posture_model: str):
    """Switch active models — only the new models are loaded."""
    inference_state.set_active_models(eye_model, posture_model)
    return {"status": "ok", "eye_model": eye_model, "posture_model": posture_model}


@app.get("/metrics")
def get_metrics():
    r = inference_state.get_last_result()
    return {
        "eye_status":     r.eye_status,
        "ear_value":      round(r.ear_value, 4),
        "posture_status": r.posture_status,
        "posture_angle":  round(r.posture_angle, 2),
        "health_score":   round(r.health_score, 1),
        "face_detected":  r.face_detected,
        "eye_model":      inference_state.eye_model_name,
        "posture_model":  inference_state.posture_model_name,
    }
