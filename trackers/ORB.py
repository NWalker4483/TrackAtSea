import numpy as np
from sklearn.cluster import DBSCAN
import cv2 

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
        self.detections = [] # [(frame, id, box)]
        self.orb = cv2.ORB_create()
        self.prev_clusters = {}
        self.Land = Land_at 
        self.__new_id = 0
        self.frames_read = 0
        
    def getAllDetections(self):
        return None
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

        clusters = {}
        for point, label in zip(pts, DBSCAN(eps=20, min_samples=4).fit_predict(pts)): # Cluster the points 
            if label == -1: continue # Skip the noise points
            if label in clusters: # Group The points by cluster
                clusters[label].append(point)
            else:
                clusters[label] = [point]

        grouped_clusters = {}
        centers = []
        for cluster_id in self.prev_clusters:
            if len(clusters) == 0: break

            centers.append(avg(self.prev_clusters[cluster_id]))
            cnt = centers[-1]
            
            best = min(clusters, key = lambda x: dist(avg(clusters[x]),cnt))
            grouped_clusters[cluster_id] = clusters[best]
            del clusters[best]
        
        for cluster_id in clusters: # Non assigned clusters
            grouped_clusters[self.__new_id] = clusters[cluster_id]
            centers.append(avg(clusters[cluster_id]))
            self.__new_id += 1 
        
        detections = []
        for cluster_id in grouped_clusters:
            x1, y1, x2, y2 = 99999, 99999, 0, 0 
            for point in grouped_clusters[cluster_id]:
                point = (int(point[0]), int(point[1]))
                x1, y1, x2, y2 = min([point[0],x1]), min([point[1],y1]), max([point[0],x2]), max([point[1],y2])
            detections.append((self.frames_read, cluster_id, (x1, y1, x2, y2)))
        self.frames_read += 1
        self.prev_clusters = grouped_clusters
        self.detections += detections
        return detections 
if __name__ == "__main__":
    import random
    import csv

    video_num = 6
    cap = cv2.VideoCapture(f'raw_data/video/{video_num}.mp4')
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    smp = cap.read()[1]
    out = cv2.VideoWriter('output.orb3.mp4', fourcc, 15,
                    (smp.shape[1],  smp.shape[0]))
    Land = 420 
    tracker = ORBTracker(Land)

    try:
        out_file = open(f"generated_data/tracks/orb.{video_num}.csv","w+")
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
                color = id_to_random_color(ID)
                cv2.rectangle(img2, (x1, y1), (x2, y2), color, 2)
                csvwriter.writerow(
                    [frame, ID, x1, y1, x2, y2])
            for ID in tracker.prev_clusters:
                color = id_to_random_color(ID)
                for point in tracker.prev_clusters[ID]:
                    cv2.circle(img2, tuple([int(i) for i in point]), 1, color, 2)
            # draw only keypoints location,not size and orientation
            cv2.imshow(" ",img2)
            out.write(img2)
            cv2.waitKey(1)
    finally:
        out_file.close()
        out.release()
        pass