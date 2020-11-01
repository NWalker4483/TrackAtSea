import cv2
import bb_utils as bb
# A Python based implementation of the algorithm described on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6928767/ s


class MVDATracker():
    def __init__(self,  init_frames=250, blur_size=21, framerate=30,
                 learning_rate=3, detecting_rate=1,
                 ignored_regions=None, min_object_width=20,
                 min_object_height=20, max_HW_ratio=6,
                 min_object_area=75, detections_per_denoising=5,
                 max_recovery_distance=20):

        self.backSub = cv2.bgsegm.createBackgroundSubtractorGSOC()

        self.init_frames = init_frames
        self.framerate = framerate
        self.blur_size = blur_size
        self.learning_rate = learning_rate
        self.detecting_rate = detecting_rate

        self.detections_per_denoising = detections_per_denoising
        self.detections_since_denoising = 0

        self.ignored_regions = ignored_regions

        self.min_object_width = min_object_width
        self.min_object_height = min_object_height
        self.max_HW_ratio = max_HW_ratio
        self.min_object_area = min_object_area
        self.max_recovery_distance = max_recovery_distance

        # {ID: [[Round_(n-1) Detections],[Round_(n) Detections]]}
        self.TB = dict()
        self.background_mask = None
        self.frames_read = 0
        self.last_states = []

    def getLandmarkVessel(self):
        pass

    def update(self, frame):  # MVDA Run Every Second
        self.frames_read += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prepped_frame = cv2.blur(gray, (self.blur_size, self.blur_size))

        if self.frames_read < self.init_frames:
            self.background_mask = self.backSub.apply(prepped_frame)
            return set()

        # Update the background model {learning_rate} times per second
        if (self.frames_read % (self.framerate // self.learning_rate) == 0):
            self.background_mask = self.backSub.apply(prepped_frame)

        # Call the SUA Fucntion   5*(30//10)
        if (self.detections_since_denoising == self.detections_per_denoising):
            self.detections_since_denoising = 0
            self.StatusUpdateAlgorithm()

        if (self.frames_read % (self.framerate//self.detecting_rate) == 0):
            self.detections_since_denoising += 1
            boxes = set()  # [(x,y,width,height),]
            contours, _ = cv2.findContours(
                self.background_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                rect = cv2.boundingRect(cnt)
                if self.ignored_regions != None:
                    if self.ignored_regions[rect[1]][rect[0]] != 0:
                        continue
                box_width, box_height = rect[2:]
                if (box_height < self.min_object_height) or (box_width < self.min_object_width):
                    continue
                if ((box_width/box_height) > self.max_HW_ratio) or ((box_height/box_width) > self.max_HW_ratio) or ((box_height*box_width) < self.min_object_area):
                    continue
                if self.contiansWater(bb.crop_to(gray, rect)):  # * Not Implemented
                    continue
                boxes.add(rect)
            boxes = bb.merge_boxes(boxes)  # Merge Overlapping Boxes

            for box in boxes:  # Assign IDs to a box s
                overlap_scores = dict({-1: -1})
                dist_scores = dict({-1: self.max_recovery_distance + 1})
                for ID in self.TB:
                    max_overlap_score = -1
                    min_distance = self.max_recovery_distance
                    min_angle_difference = self.max_recovery_distance  # WARN: Not Used

                    # Compare to the previous round of detections
                    for old_box in self.TB[ID][0]:
                        overlap_score = bb.interesection_area(box, old_box)
                        max_overlap_score = overlap_score if overlap_score > max_overlap_score else max_overlap_score

                        dist_value = bb.distance_between_centers(box, old_box)
                        min_distance = dist_value if dist_value < min_distance else min_distance

                    dist_scores[ID] = min_distance
                    overlap_scores[ID] = max_overlap_score

                best_scoring_ID = max(
                    overlap_scores, key=lambda x: overlap_scores[x])
                closest_box_ID = min(dist_scores, key=lambda x: dist_scores[x])

                if overlap_scores[best_scoring_ID] > 0:
                    self.TB[best_scoring_ID][1].append(box)
                    continue
                if dist_scores[closest_box_ID] < self.max_recovery_distance:
                    self.TB[closest_box_ID][1].append(box)
                    continue
                new_id = 0 if len(self.TB) == 0 else max(self.TB) + 1
                self.TB[new_id] = [[box], [box]]
        out = []
        for i in self.TB:
            if len(self.TB[i][1]) > 0:
                out += [(i, self.TB[i][1][j])
                        for j in range(len(self.TB[i][1]))]
        return out

    def StatusUpdateAlgorithm(self):  # Is called every 5 Seconds
        copy = self.TB.copy()
        self.last_states = []
        for ID in copy:
            if len(self.TB[ID][1]) < 3:
                del self.TB[ID]
                continue
            self.TB[ID][0], self.TB[ID][1] = self.TB[ID][1], []
            self.last_states.append((ID, self.TB[ID][0][-1]))

    def contiansWater(self, clip):  # Water Detection Algortihm
        clip = cv2.bilateralFilter(clip, 9, 75, 75)
        clip = cv2.blur(clip, (self.blur_size, self.blur_size))
        edges = cv2.Canny(clip, 100, 200)  # TODO: Change to Parameters
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if 0 < len(contours) < 3:
            lengths = []
            for cnt in contours:
                _len = cv2.arcLength(cnt)
                area = cv2.contourArea(cnt)
                if _len > 100 or area < 30:
                    continue
                lengths.append(_len)
            return (sum(lengths)/len(lengths) < 40) and (max(lengths) < 250)
        else:
            return False


if __name__ == "__main__":
    a = MVDATracker()
