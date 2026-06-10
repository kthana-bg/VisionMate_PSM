"""
monitoring_tab.py
-----------------
Live monitoring using getUserMedia + HTTP frame posting.

The browser captures webcam frames via JavaScript getUserMedia,
sends each frame as a JPEG POST to the FastAPI sidecar (/frame),
and renders the annotated result back — all over plain HTTP.

No WebRTC. No STUN. No TURN. Works on Render, Railway, Fly.io, or
any standard HTTPS host without special network configuration.

Model selection is driven by whatever the user picked in
comparison_tab.py (stored in st.session_state).
"""

import os
import sys
import time

import streamlit as st
import streamlit.components.v1 as components

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from database.db_manager import DatabaseManager


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

def _status_color(status: str, good_value: str = "Normal") -> str:
    return "#2ecc71" if status == good_value else "#e74c3c"


def _health_color(score: float) -> str:
    if score >= 75:
        return "#2ecc71"
    if score >= 50:
        return "#f39c12"
    return "#e74c3c"


def _metric_card(label: str, value: str, color: str, sub: str = ""):
    sub_html = (
        f"<p style='font-size:12px;color:#aaa;margin:2px 0 0 0;'>{sub}</p>"
        if sub else ""
    )
    st.markdown(
        f"""
        <div style="
            background:#1e2130;
            border-left:4px solid {color};
            border-radius:8px;
            padding:14px 16px;
            margin-bottom:10px;">
          <p style="font-size:11px;color:#aaa;margin:0 0 4px 0;
                    text-transform:uppercase;letter-spacing:1px;">{label}</p>
          <p style="font-size:24px;font-weight:bold;color:{color};margin:0;">{value}</p>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_api_base() -> str:
    """
    Return the FastAPI base URL reachable from the browser.

    All traffic goes through the single-port Starlette reverse proxy
    (start.py) which forwards /api/* → FastAPI internally.

    On Render: https://<your-service>.onrender.com/api
    Locally:   http://localhost:10000/api  (proxy also runs locally)

    Using a relative path (/api) means this works on any hostname
    without any environment variable configuration.
    """
    # Always use the /api prefix — the proxy in start.py handles routing.
    # window.location.origin gives the correct base in the browser JS.
    return "/api"


# ──────────────────────────────────────────────────────────
# Live-feed HTML component
# ──────────────────────────────────────────────────────────

def _build_live_component(api_base: str, eye_model: str, posture_model: str) -> str:
    """
    Returns a self-contained HTML+JS string that:
      1. Opens the webcam via getUserMedia (no WebRTC, no peer connection).
      2. Every 150 ms grabs a frame from a hidden <canvas>.
      3. POSTs it as a base64 JPEG to {api_base}/frame.
      4. Draws the annotated response back onto a visible <canvas>.
      5. Updates the on-screen metric cards.
    """
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f1117;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
  }}

  #wrapper {{
    display: flex;
    gap: 16px;
    padding: 8px;
  }}

  /* ── Video column ─────────────────── */
  #video-col {{
    flex: 2;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}

  #display-canvas {{
    width: 100%;
    border-radius: 10px;
    background: #1e2130;
    display: block;
  }}

  #status-bar {{
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 13px;
  }}

  .dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
  }}
  .dot-green  {{ background: #2ecc71; box-shadow: 0 0 6px #2ecc71; }}
  .dot-red    {{ background: #e74c3c; box-shadow: 0 0 6px #e74c3c; }}
  .dot-grey   {{ background: #555; }}

  /* ── Controls ─────────────────────── */
  .btn {{
    padding: 8px 20px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: opacity .15s;
  }}
  .btn:hover {{ opacity: .85; }}
  .btn-start  {{ background: #2ecc71; color: #0f1117; }}
  .btn-stop   {{ background: #e74c3c; color: #fff; }}
  .btn-row    {{ display: flex; gap: 10px; margin-top: 4px; }}

  /* ── Metrics column ───────────────── */
  #metrics-col {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}

  .metric-card {{
    background: #1e2130;
    border-left: 4px solid #3498db;
    border-radius: 8px;
    padding: 12px 14px;
  }}
  .metric-label {{
    font-size: 10px;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
  }}
  .metric-value {{
    font-size: 22px;
    font-weight: 700;
  }}
  .metric-sub {{
    font-size: 11px;
    color: #aaa;
    margin-top: 2px;
  }}

  #model-info {{
    background: #1e2130;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    color: #aaa;
    line-height: 1.7;
  }}

  #face-badge {{
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    text-align: center;
  }}

  #timer {{
    font-size: 12px;
    color: #aaa;
    text-align: center;
  }}
</style>
</head>
<body>
<div id="wrapper">

  <!-- ── Video column ───────────────────────── -->
  <div id="video-col">

    <canvas id="display-canvas" width="640" height="480">
      Your browser does not support canvas.
    </canvas>

    <div id="status-bar">
      <span class="dot dot-grey" id="conn-dot"></span>
      <span id="conn-label">Camera off</span>
      &nbsp;|&nbsp;
      <span id="fps-label">0 fps</span>
      &nbsp;|&nbsp;
      <span id="timer">00:00</span>
    </div>

    <div class="btn-row">
      <button class="btn btn-start" onclick="startMonitoring()">&#9654; Start</button>
      <button class="btn btn-stop"  onclick="stopMonitoring()">&#9646;&#9646; Stop</button>
    </div>

  </div>

  <!-- ── Metrics column ────────────────────── -->
  <div id="metrics-col">

    <div class="metric-card" id="card-health">
      <div class="metric-label">Health Score</div>
      <div class="metric-value" id="val-health" style="color:#3498db;">— / 100</div>
      <div class="metric-sub">Combined eye + posture</div>
    </div>

    <div class="metric-card" id="card-eye">
      <div class="metric-label">Eye Status</div>
      <div class="metric-value" id="val-eye" style="color:#3498db;">—</div>
      <div class="metric-sub" id="sub-eye">EAR: —</div>
    </div>

    <div class="metric-card" id="card-posture">
      <div class="metric-label">Posture Status</div>
      <div class="metric-value" id="val-posture" style="color:#3498db;">—</div>
      <div class="metric-sub" id="sub-posture">Neck angle: —</div>
    </div>

    <div id="model-info">
      <b>Eye model:</b> <span id="info-eye">{eye_model}</span><br>
      <b>Posture model:</b> <span id="info-posture">{posture_model}</span><br>
      Eye latency: <span id="info-eye-lat">—</span> ms<br>
      Posture latency: <span id="info-pose-lat">—</span> ms
    </div>

    <div id="face-badge" style="background:#3498db22;border:1px solid #3498db;color:#3498db;">
      Waiting for camera...
    </div>

  </div>
</div>

<!-- Hidden video + capture canvas -->
<video id="raw-video" autoplay playsinline muted
       style="display:none;width:640px;height:480px;"></video>
<canvas id="capture-canvas" width="640" height="480"
        style="display:none;"></canvas>

<script>
// ──────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────
// Build absolute URL from relative path so it works on any hostname
const _RAW_BASE      = "{api_base}";
const API_BASE       = _RAW_BASE.startsWith("/")
                         ? window.location.origin + _RAW_BASE
                         : _RAW_BASE;
const FRAME_INTERVAL = 150;   // ms between POST requests (~6-7 fps server-side)
const JPEG_QUALITY    = 0.75;

// ──────────────────────────────────────────────────────────
// State
// ──────────────────────────────────────────────────────────
let stream        = null;
let intervalId    = null;
let sessionStart  = null;
let timerInterval = null;
let frameCount    = 0;
let lastFpsTime   = Date.now();
let pendingRequest = false;   // prevents request pile-up

const rawVideo      = document.getElementById("raw-video");
const captureCanvas = document.getElementById("capture-canvas");
const displayCanvas = document.getElementById("display-canvas");
const captureCtx    = captureCanvas.getContext("2d");
const displayCtx    = displayCanvas.getContext("2d");

// ──────────────────────────────────────────────────────────
// Start / Stop
// ──────────────────────────────────────────────────────────
async function startMonitoring() {{
  if (stream) return;   // already running

  try {{
    stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }}, frameRate: {{ ideal: 30 }} }},
      audio: false,
    }});
  }} catch (err) {{
    alert("Camera access denied: " + err.message);
    return;
  }}

  rawVideo.srcObject = stream;
  await rawVideo.play();

  sessionStart  = Date.now();
  pendingRequest = false;

  // Session timer
  timerInterval = setInterval(updateTimer, 1000);

  // Frame loop
  intervalId = setInterval(sendFrame, FRAME_INTERVAL);

  setStatus(true, "Live");
}}

function stopMonitoring() {{
  if (intervalId)    {{ clearInterval(intervalId);    intervalId    = null; }}
  if (timerInterval) {{ clearInterval(timerInterval); timerInterval = null; }}
  if (stream)        {{ stream.getTracks().forEach(t => t.stop()); stream = null; }}
  rawVideo.srcObject = null;
  setStatus(false, "Camera off");
  displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  resetMetrics();
}}

// ──────────────────────────────────────────────────────────
// Frame capture → POST → render
// ──────────────────────────────────────────────────────────
async function sendFrame() {{
  if (!stream || pendingRequest) return;

  // Capture current video frame
  captureCtx.drawImage(rawVideo, 0, 0, captureCanvas.width, captureCanvas.height);
  const dataUrl  = captureCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
  const b64      = dataUrl.split(",")[1];

  pendingRequest = true;
  try {{
    const resp = await fetch(API_BASE + "/frame", {{
      method:  "POST",
      headers: {{ "Content-Type": "application/json" }},
      body:    JSON.stringify({{ image: b64 }}),
    }});

    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    renderResult(data);
    updateFps();

  }} catch (err) {{
    console.warn("Frame POST failed:", err);
  }} finally {{
    pendingRequest = false;
  }}
}}

// ──────────────────────────────────────────────────────────
// Render annotated frame + update metric cards
// ──────────────────────────────────────────────────────────
function renderResult(data) {{
  // Draw annotated frame
  const img = new Image();
  img.onload = () => displayCtx.drawImage(img, 0, 0);
  img.src    = "data:image/jpeg;base64," + data.annotated_frame;

  // Health score
  const hs    = data.health_score;
  const hc    = hs >= 75 ? "#2ecc71" : hs >= 50 ? "#f39c12" : "#e74c3c";
  setCard("card-health", "val-health", hs.toFixed(0) + " / 100", hc);

  // Eye
  const eyeOk = data.eye_status === "Normal";
  const ec    = eyeOk ? "#2ecc71" : "#e74c3c";
  setCard("card-eye",    "val-eye",     data.eye_status, ec);
  document.getElementById("sub-eye").textContent = "EAR: " + data.ear_value.toFixed(3);

  // Posture
  const posOk = data.posture_status === "Good";
  const pc    = posOk ? "#2ecc71" : "#e74c3c";
  setCard("card-posture","val-posture", data.posture_status, pc);
  document.getElementById("sub-posture").textContent =
      "Neck angle: " + data.posture_angle.toFixed(1) + " deg";

  // Model info
  document.getElementById("info-eye").textContent      = data.eye_model     || "—";
  document.getElementById("info-posture").textContent  = data.posture_model || "—";
  document.getElementById("info-eye-lat").textContent  = data.eye_latency_ms.toFixed(1);
  document.getElementById("info-pose-lat").textContent = data.posture_latency_ms.toFixed(1);

  // Face badge
  const badge = document.getElementById("face-badge");
  if (data.face_detected) {{
    badge.textContent = "✓ Face Detected";
    badge.style.background = "#2ecc7122";
    badge.style.border      = "1px solid #2ecc71";
    badge.style.color       = "#2ecc71";
  }} else {{
    badge.textContent = "✗ No Face Detected";
    badge.style.background = "#e74c3c22";
    badge.style.border      = "1px solid #e74c3c";
    badge.style.color       = "#e74c3c";
  }}
}}

// ──────────────────────────────────────────────────────────
// UI helpers
// ──────────────────────────────────────────────────────────
function setCard(cardId, valId, text, color) {{
  document.getElementById(valId).textContent  = text;
  document.getElementById(valId).style.color  = color;
  document.getElementById(cardId).style.borderLeftColor = color;
}}

function setStatus(running, label) {{
  const dot = document.getElementById("conn-dot");
  dot.className = "dot " + (running ? "dot-green" : "dot-grey");
  document.getElementById("conn-label").textContent = label;
}}

function updateFps() {{
  frameCount++;
  const now  = Date.now();
  const diff = now - lastFpsTime;
  if (diff >= 1000) {{
    document.getElementById("fps-label").textContent =
        (frameCount / (diff / 1000)).toFixed(1) + " fps";
    frameCount  = 0;
    lastFpsTime = now;
  }}
}}

function updateTimer() {{
  if (!sessionStart) return;
  const s   = Math.floor((Date.now() - sessionStart) / 1000);
  const m   = Math.floor(s / 60);
  const sec = s % 60;
  document.getElementById("timer").textContent =
      String(m).padStart(2,"0") + ":" + String(sec).padStart(2,"0");
}}

function resetMetrics() {{
  ["val-health","val-eye","val-posture"].forEach(id => {{
    const el = document.getElementById(id);
    el.textContent = "—";
    el.style.color = "#3498db";
  }});
  document.getElementById("face-badge").textContent = "Camera stopped";
  document.getElementById("fps-label").textContent  = "0 fps";
  document.getElementById("timer").textContent       = "00:00";
  sessionStart = null;
}}
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────
# DB helper (saves metrics every ~5 s)
# ──────────────────────────────────────────────────────────

def _maybe_save_metric(db: DatabaseManager, user_id: str, metrics: dict):
    last = st.session_state.get("last_metric_save", 0)
    if time.time() - last < 5:
        return
    try:
        db.log_model_comparison(
            session_id=st.session_state.get("session_id", ""),
            user_id=user_id,
            results={
                "eye":   {"C1": {
                    "fatigue_score":    metrics.get("ear_value", 0),
                    "classification":   metrics.get("eye_status", "NORMAL").upper(),
                    "latency_ms":       metrics.get("eye_latency_ms", 0),
                    "ear":              metrics.get("ear_value", 0),
                }},
                "posture": {"C2": {
                    "slouching_prob": 1.0 if metrics.get("posture_status") == "Slouching" else 0.0,
                    "status":         metrics.get("posture_status", "GOOD").upper(),
                    "latency_ms":     metrics.get("posture_latency_ms", 0),
                    "angle_y":        metrics.get("posture_angle", 0),
                }},
                "health_score":      metrics.get("health_score", 50),
                "eye_consensus":     metrics.get("eye_status", "NORMAL").upper(),
                "posture_consensus": metrics.get("posture_status", "GOOD").upper(),
            },
        )
        st.session_state["last_metric_save"] = time.time()
    except Exception as e:
        pass   # never crash the UI over a DB write


# ──────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────

def render_monitoring_tab(user_id: str, db: DatabaseManager):
    st.header("Live Monitoring")

    # Read the model selection made in comparison_tab
    eye_model_name     = st.session_state.get("active_eye_model",     "C1 - Custom CNN")
    posture_model_name = st.session_state.get("active_posture_model", "C2 - Custom LSTM")

    # Propagate selection to the inference sidecar (in-process call)
    try:
        from inference_server import inference_state
        inference_state.set_active_models(eye_model_name, posture_model_name)
    except Exception:
        pass   # sidecar not yet ready — first few seconds after deploy

    api_base = _get_api_base()

    # ── Info banner ───────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:#1e2130;border-radius:8px;padding:10px 16px;
                    margin-bottom:12px;font-size:13px;color:#aaa;">
          Using <b style="color:#3498db">{eye_model_name}</b> for eye strain &nbsp;|&nbsp;
          <b style="color:#3498db">{posture_model_name}</b> for posture &nbsp;—&nbsp;
          change models in the <em>Comparative Analysis</em> tab.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Embed the live-feed component ─────────────────────
    html = _build_live_component(api_base, eye_model_name, posture_model_name)
    components.html(html, height=560, scrolling=False)

    # ── Tips ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### Quick Tips")
    t1, t2, t3 = st.columns(3)
    t1.markdown("**20-20-20 Rule**")
    t1.caption("Every 20 min look 20 ft away for 20 seconds.")
    t2.markdown("**Ergonomic Setup**")
    t2.caption("Top of screen at eye level, back fully supported.")
    t3.markdown("**Take Breaks**")
    t3.caption("Stand and stretch every 30-60 minutes.")
