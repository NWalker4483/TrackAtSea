import argparse
import random
import pickle
import cv2
import numpy as np

from imageai.Prediction.Custom import CustomImagePrediction

def scaleROI(box, scale=2):
    X1, Y1, X2, Y2 = box
    NX1 = X1 - (X2 - X1) * (scale - 1) / 2
    NY1 = Y1 - (Y2 - Y1) * (scale - 1) / 2
    NX2 = X2 + (X2 - X1) * (scale - 1) / 2
    NY2 = Y2 + (Y2 - Y1) * (scale - 1) / 2
    return [int(NX1), int(NY1), int(NX2), int(NY2)]


def match_tracks(detections, video_name, thresh=10, scale=4):
    matched = set()
    scores = dict()
    frame_num = 0
    cap = cv2.VideoCapture(video_name)


    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("models/resnet_model.h5")
    prediction.setJsonPath("models/resnet_model_class.json")
    prediction.loadModel(num_objects=2)
    a = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in detections:
            for ID in detections[frame_num]:
                if ID in matched:
                    continue
                try:
                    box = scaleROI(detections[frame_num][ID], scale)
                    img = frame[box[1]:box[3], box[0]:box[2]]

                    cv2.imwrite("temp.1.jpg", img)
                    predictions, probabilities = prediction.predictImage("temp.1.jpg", result_count=2)
                    boat_score = probabilities[0] if predictions[0] == 'boat' else probabilities[1]
                    if boat_score >= 65:
                        a+= 1
                        # print(predictions,probabilities,a)
                        # cv2.imshow("",img)
                        # cv2.waitKey(1)
                        scores[ID] = scores[ID] + 1 if ID in scores else 1
                        if scores[ID] >= thresh:
                            matched.add(ID)
                except Exception as e:
                    pass
        frame_num += 1
    tracks = dict({-1: dict()})
    for frame_num in detections:
        for ID in detections[frame_num]:
            if ID not in tracks:
                tracks[ID] = dict()
            if ID in matched:
                tracks[-1][frame_num] = detections[frame_num][ID]
            else:
                if len(tracks[ID]) >= 3 * 20:
                    tracks[ID][frame_num] = detections[frame_num][ID]
    
    return tracks
if __name__ == '__main__':
    match_tracks(dict({0:dict({12:[0,0,500,500]})}), f"raw_data/video/{6}.mp4")