import cv2
from deepface import DeepFace

EMOJI = {
    "happy": "Happy :)",
    "sad": "Sad :(",
    "angry": "Angry >:(",
    "surprise": "Surprised :O",
    "fear": "Fearful D:",
    "disgust": "Disgusted :/",
    "neutral": "Neutral :|"
}

COLORS = {
    "happy":    (80, 200, 80),
    "sad":      (200, 100, 50),
    "angry":    (50, 50, 220),
    "surprise": (50, 200, 220),
    "fear":     (180, 80, 180),
    "disgust":  (80, 180, 180),
    "neutral":  (180, 180, 180)
}

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
print("Starting... press Q to quit")

# Store last known emotion (so screen doesn't flicker while DeepFace thinks)
last_emotion = "neutral"
last_color   = COLORS["neutral"]
frame_count  = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:

        # Only run DeepFace every 5 frames — it's slow, this keeps video smooth
        if frame_count % 5 == 0:
            try:
                # Crop just the face out of the frame
                face_crop = frame[y:y+h, x:x+w]

                # Ask DeepFace what emotion it sees
                result = DeepFace.analyze(
                    face_crop,
                    actions=["emotion"],
                    enforce_detection=False  # don't crash if face is unclear
                )

                last_emotion = result[0]["dominant_emotion"]
                last_color   = COLORS.get(last_emotion, (180, 180, 180))

            except Exception as e:
                pass  # if it fails on a frame, just keep the last result

        # Draw bounding box in emotion colour
        cv2.rectangle(frame, (x, y), (x+w, y+h), last_color, 2)

        # Show emotion label above the box
        label = EMOJI.get(last_emotion, last_emotion)
        cv2.putText(frame, label, (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, last_color, 2)

    frame_count += 1
    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()