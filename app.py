import cv2
import streamlit as st
from deepface import DeepFace
from datetime import datetime
import time

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
Real-time facial emotion recognition
</div>
""", unsafe_allow_html=True)

# ---------------- RUN LOGIC ----------------
if "started" not in st.session_state:
    st.session_state.started = False

run = st.toggle("🎥 Start Camera")

if run:
    st.session_state.started = True

if not run and not st.session_state.started:
    st.markdown("""
    <div style="text-align:center;margin-top:80px;color:#666;">
        <div style="font-size:3rem;">🎥</div>
        <div>Start the camera to begin emotion detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not run and st.session_state.started:
    st.markdown("<div style='text-align:center;color:#666;'>Camera stopped</div>", unsafe_allow_html=True)
    st.stop()

# ---------------- LAYOUT ----------------
left, right = st.columns([3, 2])

with left:
    st.markdown(
        "<div style='background:#111;padding:10px;border-radius:12px;border:1px solid #2a2a2a'>",
        unsafe_allow_html=True
    )
    frame_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    col1, col2 = st.columns(2)
    conf_card = col1.empty()
    faces_card = col2.empty()

    st.markdown("### Emotion Breakdown")
    bars = {e: st.empty() for e in [
        "Happy","Sad","Angry","Surprise","Fear","Disgust","Neutral"
    ]}

    st.markdown("### Recent Detections")
    hist_display = st.empty()

# ---------------- MODEL ----------------
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

history = []
count = 0

last = {
    "emotion": "neutral",
    "scores": {"happy":0,"sad":0,"angry":0,"surprise":0,"fear":0,"disgust":0,"neutral":0},
    "conf": 0
}

EMO_COLORS = {
    "happy": "#1D9E75",
    "sad": "#378ADD",
    "angry": "#E24B4A",
    "surprise": "#EF9F27",
    "fear": "#7F77DD",
    "disgust": "#D4537E",
    "neutral": "#888780"
}

# ---------------- LOOP ----------------
while run:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        if count % 5 == 0:
            try:
                crop = frame[y:y+h, x:x+w]
                result = DeepFace.analyze(crop, actions=["emotion"], enforce_detection=False)

                emo = result[0]["dominant_emotion"]
                scores = result[0]["emotion"]
                conf = round(max(scores.values()))

                last = {"emotion": emo, "scores": scores, "conf": conf}

                history.insert(0, {
                    "emotion": emo,
                    "conf": conf,
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                history = history[:5]

            except:
                pass

        color_hex = EMO_COLORS.get(last["emotion"], "#888888")
        rgb = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0,2,4))
        bgr = (rgb[2], rgb[1], rgb[0])

        cv2.rectangle(frame, (x,y), (x+w,y+h), bgr, 2)
        cv2.putText(frame, f"{last['emotion']} {last['conf']}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    # 🎥 SHOW CAMERA
    frame_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True
    )

    # 📊 UPDATE UI
    conf_card.markdown(f"""
    <div style="background:#1a1a1a;padding:12px;border-radius:10px;text-align:center">
        <div style="font-size:1.5rem;font-weight:700">{last['conf']}%</div>
        <div style="font-size:0.75rem;color:#666">Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    faces_card.markdown(f"""
    <div style="background:#1a1a1a;padding:12px;border-radius:10px;text-align:center">
        <div style="font-size:1.5rem;font-weight:700">{len(faces)}</div>
        <div style="font-size:0.75rem;color:#666">Faces</div>
    </div>
    """, unsafe_allow_html=True)

    for emo, placeholder in bars.items():
        pct = round(last["scores"].get(emo.lower(), 0))
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

    if history:
        hist_html = ""
        for h in history:
            hist_html += f"""
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span>{h['emotion']}</span>
                <span>{h['conf']}%</span>
            </div>
            """
        hist_display.markdown(hist_html, unsafe_allow_html=True)

    count += 1
    time.sleep(0.03)

cap.release()