import argparse
import random
import pickle 
import cv2
import numpy as np
import keras
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
    cap = cv2.VideoCapture(video_name)
    frame_num = 0
    model = keras.models.load_model("models/model.h5")
    shape = model.layers[0].input_shape[0]
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_num in detections:
            for ID in detections[frame_num]:
                if ID in matched: continue 
                try:
                    box = scaleROI(detections[frame_num][ID], scale)
                    img = frame[box[1]:box[3], box[0]:box[2]]
                    img = cv2.resize(img,(shape[1],shape[2]))
                    img = np.reshape(img,[1,shape[1],shape[2],3])
                    det = model.predict(img)[0][0]
                    if det > .5:
                        print(000000)
                        scores[ID] = scores[ID] + 1 if ID in scores else 1
                        if scores[ID] >= thresh:
                            print(111111)
                            matched.add(ID)
                except Exception as e:
                    pass#print(str(e))
                    
        frame_num += 1
    tracks = dict({-1: dict()})
    for frame_num in detections:
        for ID in detections[frame_num]:
            if ID not in tracks: tracks[ID] = dict()
            if ID in matched:
                tracks[-1][frame_num] = detections[frame_num][ID]
            else:
                tracks[ID][frame_num] = detections[frame_num][ID]
    return tracks


         