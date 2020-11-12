import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from sort.sort import Sort
import numpy as np
import cv2
from trackers.ORB import ORBTracker
import random
import csv 
from utils.common import * 
#create instance of SORT
mot_tracker = Sort()
orb_tracker = ORBTracker() 

# get detections
videonum = 14
cap = cv2.VideoCapture(f'raw_data/video/{videonum}.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
smp = cap.read()[1]
out = cv2.VideoWriter('output.sort.mp4', fourcc, 15,
                    (smp.shape[1],  smp.shape[0]))

try:
    out_file = open(f"generated_data/tracks/sort.orb.{videonum}.csv","w+")
    fields = ['Frame No.', 'Vessel ID','X1','Y1','X2','Y2']  
    csvwriter = csv.writer(out_file)  
    csvwriter.writerow(fields)  
    a = 0 
    while True:
        ret, frame = cap.read()
        if not ret: break

        detections = orb_tracker.update(frame)
        # Convert to SOrt Format
        updates = []
        for detection in detections:
            _, _, box = detection
            box = list(box)
            box.append(1)
            updates.append(box)
        # update SORT
        if len(updates) > 0:
            track_bbs_ids = mot_tracker.update(np.array(updates))
        else: 
            track_bbs_ids = mot_tracker.update()

        # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
        for detection in track_bbs_ids:
            x, y, x2, y2, ID = [int(i) for i in detection]
            cv2.rectangle(frame, (x, y), (x2, y2), id_to_random_color(ID), 2)
            csvwriter.writerow([a, ID, x, y, x2, y2])
        out.write(frame)
        cv2.imshow('frame', frame)
        a+=1
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
finally:
    out_file.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
