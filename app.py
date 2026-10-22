"""
Smart Face Recognition Attendance System - Streamlit Web Application
Run from terminal or VS Code:
    streamlit run app.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.camera import CameraStream
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.attendance_manager import AttendanceManager
from core.trainer import ModelTrainer
from core.config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_IP_CAM_URL,
    LBPH_CONFIDENCE_THRESHOLD,
    FACE_IMG_SIZE,
    PROJE_DIR,
    ATTENDANCE_CSV,
    ATTENDANCE_DB
)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="AI Face Attendance System",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-badge-green {
        color: #065F46;
        background-color: #D1FAE5;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .status-badge-red {
        color: #991B1B;
        background-color: #FEE2E2;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SINGLETON RESOURCES -----------------
@st.cache_resource
def get_detector():
    return FaceDetector()


def get_recognizer(threshold):
    return FaceRecognizer(threshold=threshold)


def get_attendance_manager():
    return AttendanceManager()


detector = get_detector()
attendance_mgr = get_attendance_manager()

# ----------------- SIDEBAR SETTINGS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/facial-recognition.png", width=70)
    st.title("System Control")

    st.subheader("📷 Camera Settings")
    cam_type = st.radio("Camera Source", ["USB Webcam", "IP Camera / Phone URL"], index=0)

    if cam_type == "USB Webcam":
        cam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
        active_source = int(cam_idx)
    else:
        ip_url = st.text_input("IP Camera URL", value=DEFAULT_IP_CAM_URL)
        active_source = ip_url.strip()

    st.divider()

    st.subheader("⚙️ Recognition Parameters")
    conf_threshold = st.slider(
        "LBPH Distance Threshold",
        min_value=40.0,
        max_value=120.0,
        value=float(LBPH_CONFIDENCE_THRESHOLD),
        step=5.0,
        help="Lower distance means stricter matching. Faces exceeding this distance are marked 'Unknown'."
    )

    st.divider()
    recognizer = get_recognizer(conf_threshold)

    st.subheader("👥 System Status")
    if recognizer.is_loaded:
        registered = recognizer.get_registered_persons()
        st.markdown(f"**Status:** <span class='status-badge-green'>Model Ready</span>", unsafe_allow_html=True)
        st.write(f"**Registered Users ({len(registered)}):**")
        for u in registered:
            st.write(f"• {u}")
    else:
        st.markdown(f"**Status:** <span class='status-badge-red'>Model Not Loaded</span>", unsafe_allow_html=True)
        st.info("Register a person or train the model in the 'Register & Train' tab.")

# ----------------- APP TABS -----------------
st.markdown("<div class='main-title'>Smart Face Recognition Attendance System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-time automated facial detection, recognition, and attendance logging</div>", unsafe_allow_html=True)

tab_live, tab_register, tab_records, tab_analytics = st.tabs([
    "📸 Live Attendance",
    "👤 Register & Train",
    "📊 Attendance Records",
    "📈 Analytics & Reports"
])


# =========================================================
# TAB 1: LIVE ATTENDANCE TRACKER
# =========================================================
with tab_live:
    col_left, col_right = st.columns([2.5, 1])

    with col_right:
        st.subheader("Today's Overview")
        today_summary = attendance_mgr.get_today_summary()
        st.metric("Total Present Today", today_summary["total_present"])
        st.write(f"**Date:** {today_summary['date']}")

        st.subheader("Recent Check-Ins")
        if not today_summary["records"].empty:
            recent_df = today_summary["records"][["Name", "Entry_Time", "Exit_Time"]].head(5)
            st.dataframe(recent_df, hide_index=True, use_container_width=True)
        else:
            st.info("No check-ins logged yet today.")

    with col_left:
        st.subheader("Live Camera Stream")
        run_camera = st.toggle("Activate Attendance Camera", value=False)

        frame_window = st.image([])
        status_placeholder = st.empty()

        if run_camera:
            if not recognizer.is_loaded:
                st.error("Please train or load a model first from the 'Register & Train' tab!")
            else:
                cam = CameraStream(source=active_source)
                if not cam.is_opened():
                    st.error(f"Unable to connect to camera source: `{active_source}`. Please verify your camera.")
                else:
                    status_placeholder.success("Camera is active. Looking for faces...")
                    last_notification = ""

                    try:
                        while run_camera:
                            ret, frame = cam.read()
                            if not ret or frame is None:
                                time.sleep(0.05)
                                continue

                            display_frame = frame.copy()
                            faces = detector.detect_faces(frame)

                            for bbox in faces:
                                face_crop = detector.extract_face(frame, bbox)
                                if face_crop is not None:
                                    name, conf, is_known = recognizer.predict(face_crop)

                                    if is_known:
                                        box_color = (0, 200, 0)
                                        log_res = attendance_mgr.mark_attendance(name)
                                        if log_res["status"] in ("new", "updated"):
                                            last_notification = log_res["message"]
                                    else:
                                        box_color = (0, 0, 255)

                                    detector.draw_styled_box(
                                        display_frame,
                                        bbox,
                                        label=name,
                                        color=box_color,
                                        confidence=conf
                                    )

                            # Overlay HUD
                            today_summary = attendance_mgr.get_today_summary()
                            cv2.putText(
                                display_frame,
                                f"Present: {today_summary['total_present']} | Press Stop in UI",
                                (20, 35),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA
                            )

                            if last_notification:
                                cv2.putText(
                                    display_frame,
                                    last_notification,
                                    (20, display_frame.shape[0] - 25),
                                    cv2.FONT_HERSHEY_DUPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    1,
                                    cv2.LINE_AA
                                )

                            # Streamlit expects RGB
                            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                            frame_window.image(rgb_frame, channels="RGB", use_container_width=True)

                    finally:
                        cam.release()
                        status_placeholder.info("Camera deactivated.")


# =========================================================
# TAB 2: REGISTER NEW USER & TRAIN MODEL
# =========================================================
with tab_register:
    st.subheader("Register a New Person")
    st.markdown("Add a new person to the system by capturing face samples or uploading images, then retrain the model.")

    col_reg1, col_reg2 = st.columns([1.5, 1])

    with col_reg1:
        new_name = st.text_input("Person Full Name", placeholder="e.g. Sarah Connor").strip()
        num_samples = st.slider("Number of samples to capture", min_value=30, max_value=150, value=100, step=10)

        capture_btn = st.button("📸 Capture Face Samples from Camera", type="primary", disabled=not bool(new_name))

        if capture_btn:
            if not new_name:
                st.error("Please enter a valid name first.")
            else:
                cleaned_name = "".join(c for c in new_name if c.isalnum() or c in ("_", "-")).strip()
                st.info(f"Connecting to camera ({active_source}). Please look at the camera...")

                cam = CameraStream(source=active_source)
                if not cam.is_opened():
                    st.error(f"Cannot connect to camera: {active_source}")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    preview_box = st.image([])

                    collected = 0
                    output_dir = PROJE_DIR
                    output_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        while collected < num_samples:
                            ret, frame = cam.read()
                            if not ret or frame is None:
                                time.sleep(0.04)
                                continue

                            faces = detector.detect_faces(frame)
                            if len(faces) > 0:
                                largest_face = max(faces, key=lambda f: f[2] * f[3])
                                face_crop = detector.extract_face(frame, largest_face, target_size=FACE_IMG_SIZE)

                                if face_crop is not None:
                                    save_path = output_dir / f"{cleaned_name}_{collected}.jpg"
                                    cv2.imwrite(str(save_path), face_crop)
                                    collected += 1

                                    progress_bar.progress(collected / num_samples)
                                    status_text.text(f"Capturing: {collected}/{num_samples} ({collected/num_samples*100:.0f}%)")

                                    # Preview crop
                                    preview_box.image(face_crop, caption=f"Sample {collected}", width=150)

                            time.sleep(0.04)

                        st.success(f"Successfully captured {collected} samples for **{cleaned_name}**!")

                        # Automatic Retrain
                        st.info("Automatically updating and training model...")
                        trainer = ModelTrainer()
                        res = trainer.train_lbph()
                        if res["success"]:
                            st.success(f"Model retrained! {res['message']}")
                            st.rerun()
                        else:
                            st.error(f"Training failed: {res['message']}")

                    finally:
                        cam.release()

    with col_reg2:
        st.subheader("Manual Retrain & Dataset Status")
        trainer = ModelTrainer()
        faces, ids, id_to_name = trainer.prepare_dataset()

        st.metric("Total Image Samples in Dataset", len(faces))
        st.metric("Total Registered People", len(id_to_name))

        if id_to_name:
            st.write("**People in Dataset:**")
            for pid, pname in id_to_name.items():
                count = sum(1 for i in ids if i == pid)
                st.write(f"- **{pname}**: {count} photos")

        if st.button("⚡ Force Retrain All Models"):
            with st.spinner("Training model on all dataset images..."):
                res = trainer.train_lbph()
                if res["success"]:
                    st.success(f"{res['message']}")
                    st.rerun()
                else:
                    st.error(f"Training error: {res['message']}")


# =========================================================
# TAB 3: ATTENDANCE RECORDS & EXPORT
# =========================================================
with tab_records:
    st.subheader("Attendance Records")

    # Filter Controls
    col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 1])

    with col_f1:
        search_name = st.text_input("Filter by Name", placeholder="Type name to filter...").strip()
    with col_f2:
        date_filter = st.date_input("Filter by Date", value=None)
    with col_f3:
        st.write("")
        st.write("")
        clear_filter = st.button("Reset Filters")

    date_str = date_filter.strftime("%Y-%m-%d") if date_filter and not clear_filter else None
    name_str = search_name if search_name and not clear_filter else None

    df_records = attendance_mgr.get_all_records(date=date_str, name=name_str)

    if df_records.empty:
        st.info("No attendance records match your filter criteria.")
    else:
        st.dataframe(df_records, use_container_width=True, hide_index=True)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = df_records.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=f"attendance_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )


# =========================================================
# TAB 4: ANALYTICS & INSIGHTS
# =========================================================
with tab_analytics:
    st.subheader("Attendance Analytics & Insights")

    all_df = attendance_mgr.get_all_records()

    if all_df.empty:
        st.info("Not enough data to generate analytics yet. Start by logging some attendance!")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records Logged", len(all_df))
        m2.metric("Total Unique Attendees", all_df["Name"].nunique())
        m3.metric("Average Duration (Hrs)", f"{all_df['Duration_Hours'].mean():.2f}")
        m4.metric("Days with Recorded Attendance", all_df["Date"].nunique())

        st.divider()

        col_ch1, col_ch2 = st.columns(2)

        with col_ch1:
            st.subheader("Daily Attendance Count")
            daily_counts = all_df.groupby("Date")["Name"].count().reset_index()
            daily_counts.columns = ["Date", "Attendance Count"]
            st.bar_chart(daily_counts.set_index("Date"), use_container_width=True)

        with col_ch2:
            st.subheader("Attendance by Person")
            person_counts = all_df.groupby("Name")["Date"].count().reset_index()
            person_counts.columns = ["Person", "Days Present"]
            st.bar_chart(person_counts.set_index("Person"), use_container_width=True)
