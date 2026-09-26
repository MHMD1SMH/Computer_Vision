import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance
class CamMoveEst():
    def __init__(self, frame):
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT,10,0.03)

        )
        self.min_distance = 5
        first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_gray)
        mask_features[:,0:20] = 1
        mask_features[:,-150:-1] = 1
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features,
        )

    def get_camera_movement(self, frames, read_stubs = False, path = None):
        if read_stubs and path is not None and os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
            
        camera_movement = [[0,0]*len(frames)]
        prev_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(prev_gray,**self.features)

        for frame in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame],cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, 
                                                          frame_gray,old_features,None,**self.lk_params)
            max_dist = 0
            camera_movement_x, camera_movement_y = 0,0
            for i ,(new, old) in enumerate(new_features, old_features):
                new_features_point = new.reval()
                old_features_point = new.reval()
                dist = measure_distance(new_features_point, old_features_point)
                if dist > max_dist:
                    max_dist = dist
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            if max_dist > self.min_distance:
                camera_movement[frame]= [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            prev_gray = frame_gray.copy()

        if path is not None:
            with open(path, 'wb') as f:
                pickle.dump(camera_movement,f)
        return camera_movement