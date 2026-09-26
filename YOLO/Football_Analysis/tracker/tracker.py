import cv2
from ultralytics import YOLO
import supervision as sv
import os
import pickle
import numpy as np
import sys
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball(self, ball_pos):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_pos]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_pos = [{1:{"bbox":x}}for x in df_ball_positions.to_numpy().tolist()]
        return ball_pos
    
    def detect_frames(self, frames):
        batch_size = 16
        results = []
        for i in range(0, len(frames), batch_size):
            batch_results = self.model.predict(frames[i:i + batch_size], device=0,conf = 0.1)
            results.extend(batch_results)
        return results


    def get_opject_tracks(self, frames, read_from_stub=False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detettions = self.detect_frames(frames)

        trackes = {
            'players': [],
            'referee':[],
            'ball': []
        }
        
        for frame_num, detection in enumerate(detettions):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in detection.names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object,class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object] = cls_names_inv["player"]

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            trackes['players'].append({})
            trackes['referee'].append({})
            trackes['ball'].append({})
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv["player"]:
                    trackes['players'][frame_num][track_id] = {'bbox': bbox}
                elif cls_id == cls_names_inv["referee"]:
                    trackes['referee'][frame_num][track_id] = {'bbox': bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    trackes['ball'][frame_num][1] = {'bbox': bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(trackes,f)
                
        return trackes

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame, 
                    center=(x_center, y2),
                    axes=(int(width), int(0.35*(width))),
                    angle=0, startAngle=-45, endAngle=235,
                    color=color, thickness=2,
                    lineType=cv2.LINE_4)

        rect_width = 40
        rect_height = 20
        x1_rect = x_center - rect_width // 2
        x2_rect = x_center + rect_width // 2
        y1_rect = y2 - rect_height // 2 + 15
        y2_rect = y2 + rect_height // 2 + 15

        if track_id is not None:
            cv2.rectangle(frame, 
                          (x1_rect, y1_rect), 
                          (x2_rect, y2_rect), 
                          color,
                          cv2.FILLED)
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y2 = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        triangle_height = int(0.35 * width)

        pt1 = (x_center, y2)
        pt2 = (x_center - 10, y2 - 20)
        pt3 = (x_center +10, y2 - 20)

        cv2.drawContours(frame, [np.array([pt1, pt2, pt3])], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [np.array([pt1, pt2, pt3])], 0, (0, 0, 0), 2)
        return frame
        
    def draw_annotations(self, frames, tracks):
        annotated_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            for track_id, player in tracks['players'][frame_num].items():
                color = player.get("team_color", (0, 255, 0))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))

            for _, referee in tracks['referee'][frame_num].items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 0, 255))

            for _, ball in tracks['ball'][frame_num].items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))
                
            annotated_frames.append(frame)
        return annotated_frames
