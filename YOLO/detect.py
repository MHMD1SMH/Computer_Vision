import cv2
import numpy as np
from ultralytics import YOLO
import sys

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

cap = cv2.VideoCapture(s)
model = YOLO("yolo26n.pt")

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, iou=0.5, imgsz=640,verbose=False)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)

    #cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(1)
    
    if key == 27:  
        break   

cap.release()
cv2.destroyAllWindows()