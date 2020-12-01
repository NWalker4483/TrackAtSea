from trackers.DEEP import DEEPTracker
import argparse
import random
import pickle 
import cv2
import csv 

def scaleROI(box, scale = 2):
    X1, Y1, X2, Y2 = box 
    NX1 = X1 - (X2 - X1) * (scale - 1)/ 2 
    NY1 = Y1 - (Y2 - Y1) * (scale - 1)/ 2
    NX2 = X2 + (X2 - X1) * (scale - 1)/ 2
    NY2 = Y2 + (Y2 - Y1) * (scale - 1)/ 2
    return [int(NX1), int(NY1), int(NX2), int(NY2)]

def match_tracks(detections, video_name, thresh = 3, scale = 4):
    matched = set()
    scores = dict()
    frame_num = 0
    tracker = DEEPTracker()
    cap = cv2.VideoCapture(video_name)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_num in detections:
            for ID in detections[frame_num]:
                if ID in matched: continue 
                box = scaleROI(detections[frame_num][ID], scale)
                det = tracker.update(frame[box[1]:box[3], box[0]:box[2]]) # Scale and Crop
                if len(det) != 0:
                    scores[ID] = scores[ID] + 1 if ID in scores else 1
                    if scores[ID] >= thresh:
                        matched.add(ID)
        frame_num += 1
    tracks = dict()
    for frame_num in detections:
        for ID in detections[frame_num]:
            if ID not in tracks: tracks[ID] = dict()
            if ID in matched:
                tracks[-1][frame_num] = detections[frame_num][ID]
            else:
                tracks[ID][frame_num] = detections[frame_num][ID]
    return tracks


         