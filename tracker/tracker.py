from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import cv2
import sys
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # interpolate ball
    def interpolate_ball(self,ball_pos):
        ball_pos = [x.get(1,{}).get('bbox',[]) for x in ball_pos]
        df_ball_pos = pd.DataFrame(ball_pos, columns=['x1','y1','x2','y2'])

        df_ball_pos = df_ball_pos.interpolate()
        df_ball_pos = df_ball_pos.bfill()

        ball_pos = [{1:{'bbox':x}} for x in df_ball_pos.to_numpy().tolist()]

        return ball_pos

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch # += Extend medthod (append)
        return detections

    # read_from_stub=False  run def from scrath
    def get_obj_track(self, frames, read_from_stub=False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f: #'rb' = readbinary, 'wb'= writebinary
                tracks = pickle.load(f) # f = the opened file object
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players":[], #{0:{'box':{0,0,0,0}}, 1:{'box':{0,0,0,0}, ....} for one frame
                      "referees":[], #{0:{'box':{0,0,0,0}}, 1:{'box':{0,0,0,0}, ....} for one frame
                      "ball":[]} #{0:{'box':{0,0,0,0}}}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()} # key, value

            # convert to sv.detection format
            detection_sv = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper:1 => player:2
            for object_ind, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[object_ind] = cls_names_inv["player"]

            # track objects
            track_detection = self.tracker.update_with_detections(detection_sv)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in track_detection:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            # loop ball without track_id
            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}


        # save as pickle will save the result, no need to run script again.
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rec = x_center - rectangle_width//2   
        x2_rec = x_center + rectangle_width//2 
        y1_rec = (y2 - rectangle_height//2) + 15 # + 15 for random buffer
        y2_rec = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rec),(y1_rec)),
                (int(x2_rec),(y2_rec)),
                color,
                cv2.FILLED
                          )

            x1_text = x1_rec + 12  # position x1_rec + 12 pixels
            if track_id > 99: # for number more than 3 char(99+) strat a bit left of center
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rec+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (0,0,0),
                thickness = 2
            )

        return frame

    # triangle for ball spot
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x , _ = get_center_of_bbox(bbox)

        triangle_point = np.array([
            [x , y],
            [x+10 , y-15],
            [x-10 , y-15]
        ])
        cv2.drawContours(frame, [triangle_point], 0, color, cv2.FILLED) # color inside
        cv2.drawContours(frame, [triangle_point], 0, (0,0,0), 2) # Border

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))
                    
            
            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw Ball
            for track_id, ball in ball_dict.items():
                  frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))

            output_video_frames.append(frame)
        
        return output_video_frames
    
    