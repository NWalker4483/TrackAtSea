import cv2

def drawCrossHairs(pos, img):
    cv2.line(img, (pos[0] - 20, pos[1]), (pos[0] + 20, pos[1]), (0, 0, 255), 1)
    cv2.line(img, (pos[0], pos[1] - 20), (pos[0], pos[1] + 20), (0, 0, 255), 1)
    return img



class ManualTracker():
    def __init__(self, frame_count, num_detections):
        self.pos = [400, 400]  # x, y
        self.pos2 = [400, 400]  # x, y
        self.frame_count = frame_count
        self.num_detections = num_detections
        self.last_frame = None
        cv2.namedWindow("Tracker Frame")
        self.frames_read = 0 

    def getLandmarkVessel(self):
        key = None
        while key != ord("f"):
            temp = self.last_frame.copy()
            cv2.rectangle(temp, (self.pos[0], self.pos[1]), (self.pos2[0], self.pos2[1]), (255, 0, 0), 2)
            key = cv2.waitKey(1) & 0xFF
            cv2.imshow("Tracker Frame",temp)
            if key == ord('d'):
                return None
        x1 = min(self.pos[0],self.pos2[0]) 
        y1 = min(self.pos[1],self.pos2[1]) 
        x2 = max(self.pos[0],self.pos2[0]) 
        y2 = max(self.pos[1],self.pos2[1])
        return (x1, y1, x2, y2)
    
    def update(self,frame):
        det = [] 
        self.last_frame = frame 
        if self.frames_read % (self.frame_count // self.num_detections) == 0:
            loc = self.getLandmarkVessel()
            if loc != None: 
                det = [(self.frames_read,-1,loc)]
        self.frames_read += 1
        return det
        # key = cv2.waitKey(1) & 0xFF
        # cv2.imshow("Tracker Frame", self.last_frame)
        
if __name__ == "__main__":
    import argparse
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

    video = cv2.VideoCapture(args.video_file)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_read = 0 

    def drag(event, x, y, flags, param): 
        global dragging, tracker    
        if dragging:
            tracker.pos2 = [x,y] 
        if event == cv2.EVENT_LBUTTONDOWN:
            tracker.pos = [x,y] 
            dragging = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    cv2.namedWindow("Tracker Frame")
    cv2.setMouseCallback("Tracker Frame", drag)
    tracker = ManualTracker(frame_count, args.num_detections)
    dragging = False
    try:
        out_file = open(f"generated_data/tracks/manual.{args.video_num}.csv","w+")
        fields = ['Frame No.', 'Vessel ID','X1','Y1','X2','Y2']  
        csvwriter = csv.writer(out_file)  
        csvwriter.writerow(fields) 
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break
            detection = tracker.update(frame)
            if len(detection) != 0: 
                f_id, ID, box = detection[0]
                x1, y1, x2, y2 = box
                csvwriter.writerow(
                    [f_id, ID, x1, y1, x2, y2])
    finally: 
        out_file.close()