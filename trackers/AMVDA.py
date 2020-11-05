import cv2
import bb_utils as bb
import imageai
from trackers.MVDA import MVDATracker
# Aided moving vehicle detection algorithm
class AMVDATracker():
    def __init__(self,  scaling_factor=3, init_frames=250,blur_size=21, framerate=30,
                 learning_rate=3, detecting_rate=1,
                 ignored_regions=None, min_object_width=20,
                 min_object_height=20, max_HW_ratio=6,
                 min_object_area=75, detections_per_denoising=5,
                 max_recovery_distance=20):
        self.mvda = MVDATracker(init_frames, blur_size, framerate,\
            learning_rate, detecting_rate,\
            ignored_regions, min_object_width,\
            min_object_height, max_HW_ratio,\
            min_object_area, detections_per_denoising,\
            max_recovery_distance)
        self.frames_per_round = 0 
        self.frames_in_round = []
        self.scaling_factor = scaling_factor
        self.last_states = []
        self.detections = []
        
    def getAllDetections(self):
        return self.detections

    def getLandmarkVessel(self):
        # Return the oldest set off boxes
        self.mvda 
        pass

    def update(self, frame):  # MVDA Run Every Second
        # Before the status update algorithm is run 
        if (self.mvda.detections_since_denoising == self.mvda.detections_per_denoising):
            copy = self.mvda.TB.copy()
            self.mvda.last_states = []
            for ID in copy:
                if len(self.mvda.TB[ID][1]) < 3:
                    del self.mvda.TB[ID]
                    continue
                self.mvda.TB[ID][0], self.mvda.TB[ID][1] = self.mvda.TB[ID][1], []
                self.mvda.last_states.append((ID, self.mvda.TB[ID][0][-1]))
            self.frames_in_round = []
        # On Every detection frame
        if (self.mvda.frames_read % (self.mvda.framerate//self.mvda.detecting_rate) == 0):
            self.frames_in_round.append(frame)

        assert(self.mvda.frames_read== 0 )
        self.mvda.update(frame)
        pass



if __name__ == "__main__":
    pass
