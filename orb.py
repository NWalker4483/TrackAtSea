import numpy as np
import cv2 
from matplotlib import pyplot as plt
import random

def drawCrossHairs(pos, img):
    cv2.line(img, (pos[0] - 10, pos[1]), (pos[0] + 10, pos[1]), (0, 0, 255), 1)
    cv2.line(img, (pos[0], pos[1] - 10), (pos[0], pos[1] + 10), (0, 0, 255), 1)
    return img

cache = {}
def id_to_random_color(number):
    if not number in cache:
        r, g, b = random.randint(0,255),random.randint(0,255),random.randint(0,255)
        cache[number]= (r, g, b)
        return r, g, b
    else:
        return cache[number]

def dist(p1,p2):
    return (((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)) ** .5
    
def avg(c):
    x = 0 
    y = 0 
    for point in c:
        x += point[0]
        y += point[1]
    return (x/len(c),y/len(c))

cap = cv2.VideoCapture('raw_data/video/6.mp4')
from sklearn.cluster import DBSCAN
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
smp = cap.read()[1]
out = cv2.VideoWriter('output.orb.mp4', fourcc, 15,
                   (smp.shape[1],  smp.shape[0]))
# Initiate ORB detector
orb = cv2.ORB_create()
last_clusters = {}
Land = 420 
from filterpy.kalman import KalmanFilter
kf = KalmanFilter(dim_x=2, dim_z=1)

# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb
try:
    cluster_kfs = {}
    while(1):
        ret, img = cap.read()
        if not ret: exit()
        
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        cv2.line(img, (0, Land), (img.shape[1], Land), (0, 0, 255), 2) # Ignore the land
        pts = []
        for point in kp:
            if point.pt[1] > Land: 
                pts.append(point.pt)
        pts = np.array(pts)
        img2 = img.copy()

        clusters = {}
        centers = []
        for point, label in zip(pts, DBSCAN(eps=20, min_samples=4).fit_predict(pts)):
            if label == -1: continue 
            if label in clusters:
                clusters[label].append(point)
            else:
                clusters[label] = [point]
                # cx, cy, x1, y1, x2, y2, dx1, dy1, dx2, dy2 
                cluster_kfs[label] = KalmanFilter(dim_x=4, dim_z=1) 
                cluster_kfs[label].x = np.array([2., 0.])
                f.R = 5
                cluster_kfs[label].P *= 1000.
        # Create Kalman for each cluster
        
        new_clusters = {}
        for cluster_id in last_clusters:
            cnt = avg(last_clusters[cluster_id])
            centers.append(cnt)
            if len(clusters) == 0: break   # ! NOT A FIX 
            best = min(clusters, key = lambda x: dist(avg(clusters[x]),cnt))
            new_clusters[cluster_id] = clusters[best]
            del clusters[best]
        
        new_id = max(last_clusters) + 1 if len(last_clusters) != 0 else 0 
        for cluster_id in clusters: # Non assigned clusters
            new_clusters[new_id] = clusters[cluster_id]
            centers.append(avg(clusters[cluster_id]))
            new_id += 1 
        ##############################
        for cluster_id in new_clusters:
            color = id_to_random_color(int(cluster_id))
            bboxes = []
            x1, y1, x2, y2 = 9999, 9999, 0, 0 
            for point in new_clusters[cluster_id]:
                point = (int(point[0]), int(point[1]))
                x1, y1, x2, y2 = min([point[0],x1]), min([point[1],y1]), max([point[0],x2]), max([point[1],y2])
                img2 = cv2.circle(img2, point, 3, color, -1)
            cv2.rectangle(img2, (x1, y1), (x2, y2), (0,255,0), 2)

        for point in centers:
            point = (int(point[0]), int(point[1]))
            img2 = drawCrossHairs(point, img2)
        last_clusters = new_clusters

        # draw only keypoints location,not size and orientation
        cv2.imshow(" ",img2)
        out.write(img2)
        cv2.waitKey(1)
finally:
    out.release()
    pass