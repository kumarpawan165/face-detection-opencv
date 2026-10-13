import cv2

# Pre-trained Haar Cascade face model that ships with OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Preprocessing: grayscale is faster and colour is not needed for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scaleFactor=1.3, minNeighbors=5 (try 1.1 and 2 to see the difference)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
