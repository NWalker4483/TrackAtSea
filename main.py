import argparse
import cameratransform as ct
import pickle
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_file', type=str, default="raw_data/video/6.mp4",
                    help='Video file name (MP4 format)')
parser.add_argument('--num_detections', type=int, default=25,
                    help='')
parser.add_argument('--camera_file_path', type=str, default="generated_data/frame2gps/frame2gps.6.list",
                    help='')
parser.add_argument('--cam_lat', type=float, default = 32.70297, 
                    help='Latitude of source in decimal degrees (i.e. where the camera is mounted')
parser.add_argument('--cam_long', type=float, default = -117.23463100000001,
                    help='Longitude of source in decimal degrees (i.e. where the camera is mounted')
parser.add_argument('--tracker_type', type=str, default="Manual",
                    help='')

args = parser.parse_args()

video = cv2.VideoCapture(args.video_file)
cam = ct.load_camera(args.camera_file_path)
cam.setGPSpos(args.cam_lat, args.cam_long)

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frames_read = 0 

if args.tracker_type == "MVDA":
    from MVDA import MVDATracker
    tracker = MVDATracker(init_frames=50, detecting_rate=1, detections_per_denoising=5, framerate=20, max_recovery_distance= 50)

elif args.tracker_type == "DEEP":
    pass

elif args.tracker_type == "Manual":
    from Manual import ManualTracker
    def drag(event, x, y, flags, param): 
        global dragging, tracker    
        if dragging:
            tracker.pos = [x,y] 
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
    cv2.namedWindow("Tracker Frame")
    cv2.setMouseCallback("Tracker Frame", drag)
    tracker = ManualTracker()
    dragging = False
else:
    print("Some Error Msg")
    exit()

out_file = open("out.csv","w+")
lm_points_px = [] # Boat Posisition in each frame
pairs = []
try:      
    while video.isOpened():
        ret, frame = video.read()
        if not ret: break

        tracker.update(frame)
        if frames_read % (frame_count // args.num_detections) == 0:
            x, y = tracker.getLandmarkVessel()
            pairs.append((lm_points_space[frames_read],[x, y]))
        frames_read += 1

    for space_coords, px in pairs:
        pass # Write to file 
    if True:
        #import matplotlib.pyplot as plot
        base, detected = zip(*pairs)
        def drawPath(path,img):

            path
            return img 
        
    # cam.plotTrace()
    # plt.tight_layout()
finally:
    cam.save(f"generated_data/transforms/{args.tracker_type.lower()}.ctObject") # ? maybe add a time to the filename
    out_file.close()