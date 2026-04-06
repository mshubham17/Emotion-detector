import cv2

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open your webcam (0 = default camera)
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale — Haar works on single-channel images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # THE MAGIC LINE — detect faces
    # scaleFactor: how much to shrink image at each scale
    # minNeighbors: how many overlapping detections = confirmed face
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a green rectangle around every detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Optional: label it
        cv2.putText(frame, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show face count on screen
    cv2.putText(frame, f"Faces: {len(faces)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Face Detector", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()