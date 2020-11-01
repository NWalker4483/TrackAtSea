import cv2

def drawCrossHairs(pos, img):
    cv2.line(img,(pos[0] - 20,pos[1]),(pos[0] + 20,pos[1]),(0,0,255),1)
    cv2.line(img,(pos[0],pos[1] - 20),(pos[0], pos[1] + 20),(0,0,255),1)
    return img 

class ManualTracker():
    def __init__(self):
        self.pos = [400,400] # x, y 
        self.last_frame = None
        cv2.namedWindow("Tracker Frame")
        
    def getLandmarkVessel(self):
        key = None  
        while key != ord("f"):
            key = cv2.waitKey(1) & 0xFF
            cv2.imshow("Tracker Frame", drawCrossHairs(self.pos,self.last_frame.copy()))
            if key == ord('w'):
                self.pos[1] += -5
            if key == ord('s'):
                self.pos[1] += 5
            if key == ord('a'):
                self.pos[0] += -5
            if key == ord('d'):
                self.pos[0] += 5
        return self.pos
    
    def update(self,frame):
        self.last_frame = frame 
        # key = cv2.waitKey(1) & 0xFF
        # cv2.imshow("Tracker Frame", self.last_frame)
        
    