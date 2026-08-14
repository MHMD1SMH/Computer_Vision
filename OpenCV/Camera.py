import os

import cv2
import sys

import numpy

PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

feature_params = dict(maxCorners=400, qualityLevel=0.3, minDistance=7, blockSize=7)

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

image_filter = PREVIEW

cap = cv2.VideoCapture(s)
win_name = 'Camera'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
resut = 0

while cv2.waitKey(1) != 27:  # 27 is the ASCII code for the ESC key
    has_frame, frame = cap.read()
    if not has_frame:
        break
    if image_filter == PREVIEW:
        resut = frame
    elif image_filter == BLUR:
        resut = cv2.GaussianBlur(frame, (15, 15), 0)
    elif image_filter == FEATURES:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, **feature_params)
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(resut, (int(x), int(y)), 10, (0, 255, 0), 1)
    elif image_filter == CANNY:
        resut = cv2.Canny(frame, 100, 200)
    cv2.imshow(win_name, resut)

    key = cv2.waitKey(1)
    if key == ord('b'):
        image_filter = BLUR
    elif key == ord('f'):
        image_filter = FEATURES
    elif key == ord('c'):
        image_filter = CANNY
    elif key == ord('p'):
        image_filter = PREVIEW


cap.release()
cv2.destroyWindow(win_name)