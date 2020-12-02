import argparse
import pickle
import cv2
import numpy as np 
import csv

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--video_num', type=int, default=6)

parser.add_argument('--video_file', type=str,
                    help='Video file name (MP4 format)')
parser.add_argument('--num_detections', type=int, default=15,
                    help='')
parser.add_argument('--camera_file_path', type=str, default="generated_data/best_camera_params.pkl",
                    help='')
parser.add_argument('--cam_lat', type=float, default=32.70297,
                    help='Latitude of source in decimal degrees (i.e. where the camera is mounted')
parser.add_argument('--cam_long', type=float, default=-117.23463100000001,
                    help='Longitude of source in decimal degrees (i.e. where the camera is mounted')
parser.add_argument('--tracker_type', type=str, default="ORB",
                    help='')
parser.add_argument('--detection_file', type=str, default=None,
                    help='')


args = parser.parse_args()
if args.video_file == None:
    args.video_file = f"raw_data/video/{args.video_num}.mp4"
    
video = cv2.VideoCapture(args.video_file)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if args.tracker_type == "ORB":
    from trackers.ORB import ORBTracker
    tracker = ORBTracker(Land_at=420)
elif args.tracker_type == "MANUAL":
    from trackers.MANUAL import ManualTracker
    def drag(event, x, y, flags, param):
        global dragging, tracker # Defined in fit_transfomr.py drag function 
        if dragging:
            tracker.pos = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
    cv2.namedWindow("Tracker Frame")
    cv2.setMouseCallback("Tracker Frame", drag)
    tracker = ManualTracker(frame_count, args.num_detections)
    dragging = False
# elif args.tracker_type == "SORT":
#     from trackers.SORT import SORTTracker
#     tracker = MVDATracker(init_frames=50, detecting_rate=1,
#                           detections_per_denoising=5, framerate=20, max_recovery_distance=50)
else:
    print("No tracker of the name exists")
    raise(KeyError)

detections = dict()
print("Detecting...")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    det = tracker.update(frame)
    if len(det) > 0:
        for frame_num, ID, box in det:
            if frame_num not in detections: detections[frame_num] = dict()
            detections[frame_num][ID] = box

# Determine Landmark Track
from match_tracks import match_tracks
print("Matching...")
tracks = match_tracks(detections, args.video_file)

# Load Homography and Camera Coefficients 
params = pickle.load(open("generated_data/best_camera_params.pkl","rb"))

# Generate and save results 
out_file = open(f"outputs/{args.video_num}.{args.tracker_type.lower()}.csv","w+")
fields = ['Frame No.', 'Vessel ID', 'Latitude', 'Longitude', 'X', 'Y']  
csvwriter = csv.writer(out_file)  
csvwriter.writerow(fields)  

out_file2 = open(f"outputs/{args.video_num}.{args.tracker_type.lower()}.main.csv","w+")
mainwriter = csv.writer(out_file)  
mainwriter.writerow(fields)  
try:
    print("Projecting...")
    for ID in tracks:
        distorted_points = []
        frames = []
        for frame_num in tracks[ID]:
            X1, Y1, X2, Y2 = tracks[ID][frame_num]
            X, Y = [X1+((X2 - X1)//2),Y2]
            frames.append(frame_num)
            distorted_points.append([[X],[Y]])
        if len(distorted_points) == 0: continue
        undistorted_points = cv2.undistortPoints(
                np.array(distorted_points, dtype=np.float64), params["Intrinsic Matrix"], params["Distortion Coefficients"], P=params["Intrinsic Matrix"])

        gps_projections = cv2.perspectiveTransform(undistorted_points, params["Homography"])
        gps_projections = gps_projections.reshape(-1, 2)
        for i in range(len(frames)):
            lat, lon = gps_projections[i]
            x, y = distorted_points[i][0][0], distorted_points[i][1][0]
            csvwriter.writerow([frames[i], ID, lat, lon, x, y])
            if ID == -1:
                mainwriter.writerow([frames[i], ID, lat, lon, x, y])
finally:
    out_file.close()
    out_file2.close()
