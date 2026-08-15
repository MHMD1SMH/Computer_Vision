import cv2
import sys

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

cap = cv2.VideoCapture(s)

win_name = "Camera"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# YuNet face detector
detector = cv2.FaceDetectorYN.create(
    "/home/mohamed-sameh/Computer_Vision/OpenCV/face_detection.onnx",
    "",
    (320, 320),
    0.7,
    0.3,
    5000
)

while True:
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    has_frame, frame = cap.read()

    if not has_frame:
        break

    # YuNet needs the actual frame size
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            # Face bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Face confidence
            confidence = face[-1]

            label = f"{confidence * 100:.2f}%"

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    cv2.imshow(win_name, frame)

cap.release()
cv2.destroyWindow(win_name)