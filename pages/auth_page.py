"""
auth_page.py  — Login / Register using st.camera_input (one-shot face capture).
Compatible with Render (no webcam stream needed for auth).
"""

import streamlit as st
import cv2
import numpy as np
import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from database.db_manager import DatabaseManager


def _decode(img_file) -> np.ndarray:
    arr = np.frombuffer(img_file.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _get_embedding(bgr):
    try:
        import face_recognition
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        locs  = face_recognition.face_locations(small, model="hog")
        if not locs:
            return None
        locs_full = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locs]
        encs      = face_recognition.face_encodings(rgb, locs_full)
        return np.array(encs[0]) if encs else None
    except ImportError:
        # face_recognition not installed — use dummy embedding
        return np.zeros(128, dtype=np.float64)


def render_auth_page(db: DatabaseManager, on_login):
    st.markdown('<div class="main-header">VisionMate</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;color:rgba(255,255,255,0.7);">'
        'Real-time Eye Strain and Posture Coach</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_login, col_reg = st.columns(2, gap="large")

    # ── Login ────────────────────────────────────────
    with col_login:
        st.markdown("#### Login")
        login_img = st.camera_input("Look at camera to log in", key="login_cam")

        if st.button("Login with Face", type="primary",
                     use_container_width=True, key="login_btn"):
            if login_img is None:
                st.warning("Please capture a photo first.")
            else:
                frame = _decode(login_img)
                emb   = _get_embedding(frame)
                if emb is None:
                    st.error("No face detected. Try better lighting.")
                else:
                    users = db.get_all_users()
                    best_match, best_score = None, 0.0
                    for u in users:
                        stored = np.array(u["face_embedding"])
                        # cosine similarity
                        n1 = np.linalg.norm(emb)
                        n2 = np.linalg.norm(stored)
                        if n1 > 0 and n2 > 0:
                            score = float(np.dot(emb, stored) / (n1 * n2))
                        else:
                            score = 0.0
                        if score > best_score and score > 0.75:
                            best_score = score
                            best_match = u
                    if best_match:
                        on_login(best_match)
                    else:
                        st.error("Face not recognised. Please register first.")

    # ── Register ─────────────────────────────────────
    with col_reg:
        st.markdown("#### Create Account")
        name    = st.text_input("Full Name", placeholder="Enter your full name")
        reg_img = st.camera_input("Capture your face", key="reg_cam")

        if st.button("Complete Registration", type="primary",
                     use_container_width=True, key="reg_btn"):
            if not name:
                st.warning("Please enter your full name.")
            elif reg_img is None:
                st.warning("Please capture a photo.")
            else:
                frame = _decode(reg_img)
                emb   = _get_embedding(frame)
                if emb is None:
                    st.error("No face detected. Try better lighting.")
                else:
                    users = db.get_all_users()
                    duplicate = False
                    for u in users:
                        stored = np.array(u["face_embedding"])
                        n1 = np.linalg.norm(emb); n2 = np.linalg.norm(stored)
                        score = float(np.dot(emb, stored) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0
                        if score > 0.75:
                            st.error(
                                f"This face is already registered as '{u['user_name']}'. "
                                "Please login instead."
                            )
                            duplicate = True
                            break
                    if not duplicate:
                        db.create_user(name, emb.tolist())
                        st.success("Registration successful. You can now log in.")
