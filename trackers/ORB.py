import numpy as np
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import cv2
import utils.common as bb
from queue import Queue
from sklearn.cluster import DBSCAN


def dist(p1,p2):
    return (((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)) ** .5
    
def avg(c):
    x = 0 
    y = 0 
    for point in c:
        x += point[0]
        y += point[1]
    return (x/len(c),y/len(c))

class ORBTracker:
    def __init__(self,Land_at=420):
        #self.detections = [] # [(frame, id, box)]
        self.orb = cv2.ORB_create()
        self.prev_clusters = {}
        self.Land = Land_at 
        self.__new_id = 0
        self.frames_read = 0
        self.frame_buffer = [] 
        
    def getLandmarkVessel(self):
        return None
    def update(self, frame):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        # find the keypoints with ORB
        kp = self.orb.detect(frame, None)

        # compute the descriptors with ORB
        # kp, des = self.orb.compute(frame, kp)

        pts = np.array([point.pt for point in kp if point.pt[1] > self.Land]) # Remove points above the threshold

        clusters = dict()
        for point, label in zip(pts, DBSCAN(eps=20, min_samples=4).fit_predict(pts)): # Cluster the points 
            if label == -1: continue # Skip the noise points
            if label in clusters: # Group The points by cluster
                clusters[label].append(point)
            else:
                clusters[label] = [point]

        temp = dict()
        for i in self.prev_clusters:
            if i + 1 > 80: continue # Only search for 80 frames back 
            temp[i+1] = self.prev_clusters[i] 
        self.prev_clusters = temp
        self.prev_clusters[0] = dict()
        assigned = set() # If a cluster in a previous frame has already been matched with a cluster in the current frame add it to this set
        for frames_back in range(1, len(self.prev_clusters)):
            if len(clusters) == 0: break
            for prev_cluster_id in self.prev_clusters[frames_back]:
                if prev_cluster_id in assigned: continue # If this idea has already been matched to a cluster then skip it
                best_id = -1 
                closest_dist = np.inf 
                for cluster_id in clusters:
                    distance = dist(avg(clusters[cluster_id]), avg(self.prev_clusters[frames_back][prev_cluster_id])) # * The center of previous clusters could be calibrated and stored though I recalculated here for simplicity when reading the code
                    if (closest_dist > distance and distance <= 50):
                        best_id = cluster_id
                        closest_dist = distance
                if best_id != -1:
                    self.prev_clusters[0][prev_cluster_id] = clusters[best_id]
                    assigned.add(prev_cluster_id)
                    del clusters[best_id]
        
        for cluster_id in clusters: # Non assigned clusters
            self.prev_clusters[0][self.__new_id] = clusters[cluster_id]
            self.__new_id += 1 
        
        detections = []
        for cluster_id in self.prev_clusters[0]: 
            x1, y1, x2, y2 = np.inf, np.inf, 0, 0 
            for point in self.prev_clusters[0][cluster_id]:
                point = (int(point[0]), int(point[1]))
                x1, y1, x2, y2 = min([point[0],x1]), min([point[1],y1]), max([point[0],x2]), max([point[1],y2])
            detections.append((self.frames_read, cluster_id, (x1, y1, x2, y2)))
        self.frames_read += 1
        return detections 
if __name__ == "__main__":
    import argparse
    import random
    import pickle 
    import csv 

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video_num', type=int, required=True)

    parser.add_argument('--video_file', type=str, default="raw_data/video/7.mp4",
                        help='Video file name (MP4 format)')
    
    parser.add_argument('--num_detections', type=int, default=20,
                        help='')

    args = parser.parse_args()

    if args.video_num != None:
        args.video_file = f"raw_data/video/{args.video_num}.mp4"

    cap = cv2.VideoCapture(args.video_file)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    smp = cap.read()[1]
    out = cv2.VideoWriter('output.orb.mp4', fourcc, 15,
                    (smp.shape[1],  smp.shape[0]))
    Land = 420 
    tracker = ORBTracker(Land)

    try:
        out_file = open(f"generated_data/tracks/orb.{args.video_num}.csv","w+")
        fields = ['Frame No.', 'Vessel ID','X1','Y1','X2','Y2'] 
        csvwriter = csv.writer(out_file)  
        csvwriter.writerow(fields)  
        while True:
            ret, img = cap.read()
            if not ret: exit()
            img2 = img.copy()
            
            cv2.line(img2, (0, Land), (img2.shape[1], Land), (0, 0, 255), 2) # Ignore the land
            
            for detect in tracker.update(img):
                frame, ID, box = detect    
                x1, y1, x2, y2 = box
                color = bb.id_to_random_color(ID)
                cv2.rectangle(img2, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img2, f'{ID} Detected ', (x2+10, y1), 0, 0.3, (0, 255, 0))
                csvwriter.writerow(
                    [frame, ID, x1, y1, x2, y2])
            for ID in tracker.prev_clusters[0]:
                color = bb.id_to_random_color(ID)
                for point in tracker.prev_clusters[0][ID]:
                    cv2.circle(img2, tuple([int(i) for i in point]), 1, color, 2)
            # draw only keypoints location,not size and orientation
            cv2.imshow(" ",img2)
            out.write(img2)
            cv2.waitKey(1)
    finally:
        out_file.close()
        out.release()
        pass