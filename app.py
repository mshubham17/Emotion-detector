import threading
import time
from datetime import datetime

import av
import cv2
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 0.5rem !important;
}
button[title="View fullscreen"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
padding:6px 0;margin-bottom:6px;border-bottom:1px solid #1f1f1f;">
    <div style="font-size:1.1rem;font-weight:600;">🎭 EmotiSense</div>
    <div style="font-size:0.75rem;color:#666;">Real-time Emotion AI</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="color:#888;font-size:0.85rem;margin-bottom:10px;">
Real-time facial emotion recognition — click <b>START</b> to allow camera access from your browser
</div>
""", unsafe_allow_html=True)


# ---------------- CACHED MODEL ----------------
@st.cache_resource
def load_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


detector = load_detector()

EMO_COLORS = {
    "happy": "#1D9E75",
    "sad": "#378ADD",
    "angry": "#E24B4A",
    "surprise": "#EF9F27",
    "fear": "#7F77DD",
    "disgust": "#D4537E",
    "neutral": "#888780",
}

# ---------------- THREAD-SAFE SHARED STATE ----------------
lock = threading.Lock()
shared_state = {
    "emotion": "neutral",
    "scores": {"happy": 0, "sad": 0, "angry": 0, "surprise": 0, "fear": 0, "disgust": 0, "neutral": 0},
    "conf": 0,
    "face_count": 0,
    "history": [],
}

frame_counter = {"count": 0}


# ---------------- VIDEO CALLBACK ----------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)

    frame_counter["count"] += 1

    with lock:
        last_emotion = shared_state["emotion"]
        last_conf = shared_state["conf"]

    for (x, y, w, h) in faces:
        # Only run DeepFace every 5 frames to save CPU
        if frame_counter["count"] % 5 == 0:
            try:
                crop = img[y:y + h, x:x + w]
                # Downscale crop to 224x224 — DeepFace resizes internally anyway
                crop_resized = cv2.resize(crop, (224, 224))
                result = DeepFace.analyze(crop_resized, actions=["emotion"], enforce_detection=False)

                emo = result[0]["dominant_emotion"]
                scores = result[0]["emotion"]
                conf = round(max(scores.values()))

                with lock:
                    shared_state["emotion"] = emo
                    shared_state["scores"] = scores
                    shared_state["conf"] = conf
                    shared_state["face_count"] = len(faces)
                    shared_state["history"].insert(0, {
                        "emotion": emo,
                        "conf": conf,
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
                    shared_state["history"] = shared_state["history"][:5]

                last_emotion = emo
                last_conf = conf
            except Exception:
                pass

        color_hex = EMO_COLORS.get(last_emotion, "#888888")
        rgb = tuple(int(color_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])

        cv2.rectangle(img, (x, y), (x + w, y + h), bgr, 2)
        cv2.putText(img, f"{last_emotion} {last_conf}%",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    # Update face count even when no DeepFace analysis runs
    if frame_counter["count"] % 5 != 0:
        with lock:
            shared_state["face_count"] = len(faces)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- LAYOUT ----------------
left, right = st.columns([3, 2])

with left:
    ctx = webrtc_streamer(
        key="emotisense",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
    )

with right:
    col1, col2 = st.columns(2)
    conf_card = col1.empty()
    faces_card = col2.empty()

    st.markdown("### Emotion Breakdown")
    bars = {e: st.empty() for e in [
        "Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"
    ]}

    st.markdown("### Recent Detections")
    hist_display = st.empty()

# ---------------- LIVE UI UPDATES ----------------
if ctx.state.playing:
    while ctx.state.playing:
        with lock:
            snap = {
                "emotion": shared_state["emotion"],
                "scores": dict(shared_state["scores"]),
                "conf": shared_state["conf"],
                "face_count": shared_state["face_count"],
                "history": list(shared_state["history"]),
            }

        conf_card.markdown(f"""
        <div style="background:#1a1a1a;padding:12px;border-radius:10px;text-align:center">
            <div style="font-size:1.5rem;font-weight:700">{snap['conf']}%</div>
            <div style="font-size:0.75rem;color:#666">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

        faces_card.markdown(f"""
        <div style="background:#1a1a1a;padding:12px;border-radius:10px;text-align:center">
            <div style="font-size:1.5rem;font-weight:700">{snap['face_count']}</div>
            <div style="font-size:0.75rem;color:#666">Faces</div>
        </div>
        """, unsafe_allow_html=True)

        for emo, placeholder in bars.items():
            pct = round(snap["scores"].get(emo.lower(), 0))
            color = EMO_COLORS.get(emo.lower(), "#888888")
            placeholder.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
                <span style="width:70px;font-size:0.8rem;">{emo}</span>
                <div style="flex:1;height:8px;background:#111;border-radius:4px;">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
                </div>
                <span style="font-size:0.8rem;">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        if snap["history"]:
            hist_html = ""
            for h in snap["history"]:
                hist_html += f"""
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                    <span>{h['emotion']}</span>
                    <span>{h['conf']}%</span>
                </div>
                """
            hist_display.markdown(hist_html, unsafe_allow_html=True)

        time.sleep(0.1)
else:
    st.markdown("""
    <div style="text-align:center;margin-top:40px;color:#666;">
        <div style="font-size:3rem;">🎥</div>
        <div>Click <b>START</b> above to allow camera access and begin emotion detection</div>
    </div>
    """, unsafe_allow_html=True)