import cv2

def drawCrossHairs(pos, img):
    cv2.line(img, (pos[0] - 20, pos[1]), (pos[0] + 20, pos[1]), (0, 0, 255), 1)
    cv2.line(img, (pos[0], pos[1] - 20), (pos[0], pos[1] + 20), (0, 0, 255), 1)
    return img



class ManualTracker():
    def __init__(self):
        self.pos = [400, 400]  # x, y
        self.pos2 = [400, 400]  # x, y
        self.last_frame = None
        cv2.namedWindow("Tracker Frame")

    def getLandmarkVessel(self):
        key = None
        while key != ord("f"):
            temp = frame.copy()
            cv2.rectangle(temp, (self.pos[0], self.pos[1]), (self.pos2[0], self.pos2[1]), (255, 0, 0), 2)
            key = cv2.waitKey(1) & 0xFF
            cv2.imshow("Tracker Frame",temp)
            if key == ord('w'):
                self.pos[1] += -5
            if key == ord('s'):
                self.pos[1] += 5
            if key == ord('a'):
                self.pos[0] += -5
            if key == ord('d'):
                self.pos[0] += 5
        x = min(self.pos[0],self.pos2[0]) 
        y = min(self.pos[1],self.pos2[1]) 
        w = max(self.pos[0],self.pos2[0]) - x
        h = max(self.pos[1],self.pos2[1]) - y
        return (x, y, w, h)
    
    def update(self,frame):
        self.last_frame = frame 
        # key = cv2.waitKey(1) & 0xFF
        # cv2.imshow("Tracker Frame", self.last_frame)
        
if __name__ == "__main__":
    import argparse
    import pickle 
    import csv 

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video_num', type=int, default=None)

    parser.add_argument('--video_file', type=str, default="raw_data/video/6.mp4",
                        help='Video file name (MP4 format)')
    parser.add_argument('--gps_list_file', type=str, default="generated_data/frame2gps/frame2gps.6.list",
                         help='')
    parser.add_argument('--num_detections', type=int, default=10,
                        help='')

    args = parser.parse_args()

    if args.video_num != None:
        args.video_file = f"raw_data/video/{args.video_num}.mp4"
        args.gps_list_file = f"generated_data/frame2gps/frame2gps.{args.video_num}.list"
    
    video = cv2.VideoCapture(args.video_file)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_read = 0 

    with open(args.gps_list_file, "rb") as fp:   # Unpickling
        lm_points_gps = pickle.load(fp)

    # assert(len(lm_points_space) == frame_count)
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
    tracker = ManualTracker()
    dragging = False

    lm_points_px = [] # Boat Posisition in each frame
    pairs = []
    try:      
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break
            
            tracker.update(frame)
            if frames_read % (frame_count // args.num_detections) == 0:
                rect = tracker.getLandmarkVessel()
                pairs.append((frames_read, lm_points_gps[frames_read],rect))
            frames_read += 1
        print(*pairs, sep= "\n")
    finally:
        out_file = open(f"generated_data/outputs/manual.detections.csv","w+")
        fields = ['Frame No.', 'Vessel ID','Latitude', 'Longitude','X','Y','W','H']  
        csvwriter = csv.writer(out_file)  
        csvwriter.writerow(fields)  
        for frame, gps, rect in pairs:
            Vessel_ID = 0
            x, y, w, h = rect
            lat, lon = gps
            csvwriter.writerow(
                [frame, Vessel_ID, lat, lon, x, y, w, h])
        out_file.close()