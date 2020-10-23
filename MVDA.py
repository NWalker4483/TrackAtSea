# A Python based implementation of the algorithm described on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6928767/
import cv2
import numpy as np

def merge_boxes(boxes):
    contained = set()
    for container in boxes:
        if container not in contained:
            for box in boxes:
                if (container[0] <= box[0]) and (container[1] <= box[1]): # Higher 
                    if (container[0] + container[2] >= box[0] + box[2]) and (container[1] + container[3] >= box[1] + box[3]): # Longer and Taller
                        if box != container:
                            contained.add(box)
    return set([box for box in boxes if box not in contained])

def bb_intersection_over_union(boxA, boxB):
    boxA = list(boxA)
    boxB = list(boxB)
    # Convert from cv2 to (xy,x,y)
    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def center_of(rect):
    pass
class MVDATracker():
    def __init__(self,  init_frames = 250, blur_size = 21, framerate = 30,
                        learning_rate = 3, detecting_rate = 1,
                        ignored_regions = None, min_object_width = 50,
                        min_object_height = 40, max_HW_ratio = 7,
                        min_object_area = 150, detections_per_denoising = 5):
        
        self.backSub = cv2.bgsegm.createBackgroundSubtractorGSOC()

        self.init_frames = init_frames
        self.framerate = framerate
        self.blur_size = blur_size
        self.learning_rate = learning_rate 
        self.detecting_rate = detecting_rate
        self.detections_per_denoising = detections_per_denoising

        self.ignored_regions = ignored_regions

        self.min_object_width = min_object_width
        self.min_object_height = min_object_height
        self.max_HW_ratio = max_HW_ratio
        self.min_object_area = min_object_area

        self.TB = dict()
        self.background_mask = None
        self.frames_read = 0

        self.old_boxes = set()
    def update(self, frame):# MVDA Run Every Second 
        self.frames_read += 1
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.blur(frame,(self.blur_size,self.blur_size))

        if self.frames_read < self.init_frames:
            self.background_mask = self.backSub.apply(frame)
            return set()
        
        if (self.frames_read % (self.framerate // self.learning_rate) == 0): # Update the background model three times per second
            self.background_mask = self.backSub.apply(frame)
        
        if (self.frames_read % (self.detections_per_denoising * (self.framerate//self.detecting_rate)) == 0): # Call the SUA Fucntion 
            self.StatusUpdateAlgorithm()
        
        if (self.frames_read % (self.framerate//self.detecting_rate) == 0):
            boxes = set() # [(x,y,width,height),]
            contours, _ = cv2.findContours(self.background_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
            for cnt in contours:
                rect = cv2.boundingRect(cnt)
                if self.ignored_regions != None:  
                    if self.ignored_regions[rect[1]][rect[0]] != 0: # 255 means this area should be ignored
                        continue
                box_width, box_height = rect[2:] 
                if (box_height < self.min_object_height) or (box_width < self.min_object_width):
                    continue
                if ((box_width/box_height) > self.max_HW_ratio) or ((box_height/box_width) > self.max_HW_ratio) or ((box_height*box_width) < self.min_object_area):
                    continue
                if self.contiansWater(rect): # * Not Implemented
                    continue
                boxes.add(rect)
            boxes = merge_boxes(boxes) # Merge Overlapping Boxes # ? The paper was not clear whether or not partially overlapping or entirely overlapping boxes should be removed as of now only fully overlapping boxes will be
                
            for box in boxes:
                scores = dict({-1:0})
                for ID in self.TB:
                    max_score = -1
                    for old_box in self.TB[ID][0]:
                        score = bb_intersection_over_union(box, old_box)
                        max_score = score if score > max_score else max_score
                    scores[ID] = max_score
                best_scoring_ID = max(scores,key = lambda x: scores[x])
                if scores[best_scoring_ID] != 0:
                    self.TB[best_scoring_ID][1].append(box)
                else:
                    if False:
                        continue
                    new_id = 0 if len(self.TB) == 0 else max(self.TB) + 1
                    self.TB[new_id] = [[],[]]
                    self.TB[new_id][1].append(box) 
        out = [] 
        for i in self.TB:
            if len(self.TB[i][1]) > 0:
                out.append((i,self.TB[i][1][-1]))
        return out  
    """
    20:  Phase 3—Write the bounding box list into the Temporary Buffer:
    21:   For each bounding box:
    22:       Calculate the percentage of intersection of the bounding box with the bounding box detected in the previous frame;
    23:       Calculate the distance between the centre of the box and the boxes from the previous round;
    24:       Calculate the angle between the centre of the box and the boxes from the previous round;
    25:       Assign the current bounding box the ID number of the bounding box from the previous round with which the intersection area is the largest—if there is no such area, select the bounding box at a distance not greater than 1.5 from the current bounding box and with an angle difference not greater than the maximum value;
    26:       If the previous step failed to obtain the ID number, then create a new one;
    27:       Write the bounding box in the Temporary Buffer with the ID number.
    """
    def StatusUpdateAlgorithm(self): # Is called every 5 Seconds 
        for i in self.TB:
            self.TB[i][0], self.TB[i][1] = self.TB[i][1], []
        # copy = self.TB.copy()
        # for ID in copy:
        #     if len(self.TB[ID][1]) < 3:
        #         if len(self.TB[ID][1]) == 0:
        #             del self.TB[ID]
        #         continue
            
        
    def contiansWater(self, frame): # Water Detection Algortihm
        return False
        # img=cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # blur=cv2.bilateralFilter(img, 9, 75, 75)
        # edges=cv2.Canny(img, 100, 200)  # TODO: Change to Parameters
        # im2, contours, hierarchy=cv.findContours(
        #     edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # if len(contours) < 3:
        #     pass 


"""
Algorithm 3 Water Detection Algorithm
1:  Convert an image to grayscale, use bilateral filter, and blur the image;
2:  Detect edges using Canny edge detector and find contours;
3:  For each contour:
4:       Calculate length of the contour (using curve approximation);
5:       Calculate the contour area;
6:       When the length is less than 100 or the area is less than 30, discard the contour;
7:  Calculate average length of the remaining contours and find the longest contour;
8:  Return water when number of contours is less than 3, the maximum length is less than 250, and average length is less than 40.
"""
if __name__ == "__main__":
    a = MVDATracker()