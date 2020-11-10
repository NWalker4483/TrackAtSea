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
        pass
    def getLandmarkVessel(self):
        pass
    def update(self, frame):
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

    cache = {}
    def id_to_random_color(number):
        if not number in cache:
            r, g, b = random.randint(0,255),random.randint(0,255),random.randint(0,255)
            cache[number]= (r, g, b)
            return r, g, b
        else:
            return cache[number]

    cap = cv2.VideoCapture('raw_data/video/6.mp4')
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    smp = cap.read()[1]
    out = cv2.VideoWriter('output.orb3.mp4', fourcc, 15,
                    (smp.shape[1],  smp.shape[0]))
    Land = 420 
    tracker = ORBTracker(Land)

    try:
        while(1):
            ret, img = cap.read()
            if not ret: exit()
            img2 = img.copy()
            cv2.line(img2, (0, Land), (img2.shape[1], Land), (0, 0, 255), 2) # Ignore the land
            
            for detect in tracker.update(img):
                _, ID, box = detect    
                color = id_to_random_color(ID)
                cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]), color, 2)
            # draw only keypoints location,not size and orientation
            cv2.imshow(" ",img2)
            out.write(img2)
            cv2.waitKey(1)
    finally:
        out.release()
        pass