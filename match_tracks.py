from trackers.DEEP import DEEPTracker
from trackers.MANUAL import ManualTracker
import argparse
import random
import pickle 
import cv2
import csv 

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_num', type=int, required=True)

parser.add_argument('--video_file', type=str, default="raw_data/video/7.mp4",
                    help='Video file name (MP4 format)')

args = parser.parse_args()

if args.video_num != None:
    args.video_file = f"raw_data/video/{args.video_num}.mp4"

cap = cv2.VideoCapture(args.video_file)
         
det_frames = dict()
with open(f"generated_data/tracks/sort.orb.{args.video_num}.csv","r") as f:
    content = f.readlines()
    content = content[1:] # Skip Header
for entry in content:
    Frame_No, ID, X1, Y1, X2, Y2 = [int(i) for i in entry.split(",")]
    if int(Frame_No) in det_frames:
        det_frames[int(Frame_No)].append((ID, [X1, Y1, X2, Y2]))
    else:
        det_frames[int(Frame_No)] = [(ID,[X1, Y1, X2, Y2])]

def scaleROI(box, scale = 2):
    X1, Y1, X2, Y2 = box 
    NX1 = X1 - (X2 - X1) * (scale - 1)/ 2 
    NY1 = Y1 - (Y2 - Y1) * (scale - 1)/ 2
    NX2 = X2 + (X2 - X1) * (scale - 1)/ 2
    NY2 = Y2 + (Y2 - Y1) * (scale - 1)/ 2
    return [int(NX1), int(NY1), int(NX2), int(NY2)]

matched = set()
scores = dict()
scale = 4
thresh = 3
frame_num = 0
tracker = DEEPTracker()
while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_num in det_frames:
        for ID, box in det_frames[frame_num]:
            if ID in matched: continue 
            box = scaleROI(box, scale)
            cv2.imshow("Detecting",frame[box[1]:box[3], box[0]:box[2]])
            cv2.waitKey(1)
            detections = tracker.update(frame[box[1]:box[3], box[0]:box[2]]) # Scale and Crop
            if len(detections) != 0:
                scores[ID] = scores[ID] + 1 if ID in scores else 1
                if scores[ID] >= thresh:
                    matched.add(ID)
    frame_num += 1
print(matched)
