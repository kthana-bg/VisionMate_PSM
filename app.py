"""
app.py  —  VisionMate main entry-point
"""

import os
os.environ["MEDIAPIPE_DISABLE_GPU"]    = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["OMP_NUM_THREADS"]          = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]     = "3"

import sys
import time
import numpy as np
import cv2
import streamlit as st

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="VisionMate",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
.main-header {
    font-size: 2.2rem; font-weight: 700; text-align: center;
    background: linear-gradient(135deg, #fff, #a0a0ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from database.db_manager import DatabaseManager
from pages.auth_page      import render_auth_page
from pages.monitoring_tab import render_monitoring_tab
from pages.comparison_tab import render_comparison_tab
from pages.analytics_tab  import render_analytics_tab


# ── Session state init ─────────────────────────────────────

def _init_state():
    defaults = {
        "logged_in":            False,
        "user_id":              None,
        "user_name":            None,
        "session_id":           None,
        "session_start":        None,
        "db":                   DatabaseManager(),
        # Model selection — defaults to Custom CNN + Custom LSTM/DNN
        "active_eye_model":     "Custom CNN",
        "active_posture_model": "Custom LSTM/DNN",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Auth callbacks ─────────────────────────────────────────

def _on_login(user: dict):
    db = st.session_state.db
    st.session_state.logged_in   = True
    st.session_state.user_id     = user["user_id"]
    st.session_state.user_name   = user["user_name"]
    st.session_state.session_id  = db.start_session(user["user_id"])
    st.session_state.session_start = time.time()
    st.rerun()


def _on_logout():
    sid = st.session_state.session_id
    if sid:
        st.session_state.db.end_session(sid)
    for k in ["logged_in", "user_id", "user_name", "session_id", "session_start"]:
        st.session_state[k] = None if k != "logged_in" else False
    st.rerun()


# ── Dashboard ──────────────────────────────────────────────

def _show_dashboard():
    # Header row
    col_h, col_btn = st.columns([4, 1])
    with col_h:
        st.markdown('<div class="main-header">VisionMate Dashboard</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;color:rgba(255,255,255,0.7);">'
            'Real-time Eye Strain &amp; Posture Monitoring</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("Logout", use_container_width=True):
            _on_logout()

    # Session info
    if st.session_state.session_start:
        mins = int((time.time() - st.session_state.session_start) / 60)
        st.caption(
            f"Session active: {mins} min  |  User: {st.session_state.user_name}"
        )

    st.divider()

    tab_live, tab_compare, tab_analytics = st.tabs([
        "Live Monitor",
        "Comparative Analysis",
        "Analytics",
    ])

    with tab_live:
        render_monitoring_tab(
            user_id = st.session_state.user_id,
            db      = st.session_state.db,
        )

    with tab_compare:
        render_comparison_tab()

    with tab_analytics:
        render_analytics_tab(user_id=st.session_state.user_id)


# ── Entry point ────────────────────────────────────────────

def main():
    try:
        if st.session_state.get("logged_in"):
            _show_dashboard()
        else:
            render_auth_page(
                db        = st.session_state.db,
                on_login  = _on_login,
            )
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page.")


if __name__ == "__main__":
    main()
