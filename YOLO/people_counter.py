import cv2
from ultralytics import YOLO

VIDEO_PATH = "people.mp4"
MODEL_PATH = "yolo26s.pt"

LINE1_Y = 620
LINE2_Y = 660

MAX_SECONDS = 30

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
max_frames = int(MAX_SECONDS * fps)
frame_count = 0

previous_positions = {}
crossed = {}
counted_ids = set()

in_count = 0
out_count = 0

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        device=0,
        classes=[0]
    )

    boxes = results[0].boxes

    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy, boxes.id):
            track_id = int(track_id)

            x1, y1, x2, y2 = map(int, box)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if track_id not in previous_positions:
                previous_positions[track_id] = center_y
                crossed[track_id] = []
                continue

            previous_y = previous_positions[track_id]

            crossed_line1 = (
                previous_y < LINE1_Y <= center_y
                or previous_y > LINE1_Y >= center_y
            )

            if crossed_line1 and 1 not in crossed[track_id]:
                crossed[track_id].append(1)

            crossed_line2 = (
                previous_y < LINE2_Y <= center_y
                or previous_y > LINE2_Y >= center_y
            )

            if crossed_line2 and 2 not in crossed[track_id]:
                crossed[track_id].append(2)

            if len(crossed[track_id]) == 2 and track_id not in counted_ids:
                if crossed[track_id] == [1, 2]:
                    out_count += 1
                    counted_ids.add(track_id)
                    print(f"ID {track_id} -> OUT")

                elif crossed[track_id] == [2, 1]:
                    in_count += 1
                    counted_ids.add(track_id)
                    print(f"ID {track_id} -> IN")

            previous_positions[track_id] = center_y

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

    cv2.line(
        frame,
        (500, LINE1_Y),
        (1300, LINE1_Y),
        (0, 255, 0),
        2
    )

    cv2.line(
        frame,
        (500, LINE2_Y),
        (1300, LINE2_Y),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"IN: {in_count}  OUT: {out_count}",
        (50, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print("IN :", in_count)
print("OUT:", out_count)