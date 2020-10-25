import argparse
import cameratransform as ct
import pickle
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_file', type=str, default="data/video/6.mp4"
                    help='Video file name (MP4 format)')
parser.add_argument('--gps_list_file', type=str, default="data/frame2gps/frame2gps.6.list"
                    help='')
parser.add_argument('--cam_lat', type=float,
                    help='Latitude of source in decimal degrees (i.e. where the camera is mounted))
parser.add_argument('--cam_long', type=float,
                    help='Longitude of source in decimal degrees (i.e. where the camera is mounted))
parser.add_argument('--tracker_type', type=str, default="Manual"
                    help='')

args = parser.parse_args()

video = cv2.VideoCapture(args.video_file)
frames_read = 0 

with open(args.gps_list_file, "rb") as fp:   # Unpickling
   lm_points_gps = pickle.load(fp)

if args.tracker_type == "MVDA":
    from MVDA import MVDATracker
    tracker = MVDATracker(init_frames=50, detecting_rate=1, detections_per_denoising=5, framerate=20, max_recovery_distance= 50 )

elif args.tracker_type == "DEEP":
    pass

elif args.tracker_type == "Manual":
    pass

lm_points_px = []
try:      
    while cap.isOpened():
        ret, frame = video.read()
        tracker.update(frame)
        x, y = tracker.getLandmarkVessel()
    # ! Double check that this conversion is necessary 
    lm_points_space = cam.spaceFromGPS(lm_points_gps)
    cam.addLandmarkInformation(lm_points_px, lm_points_space, [3, 3, 5])
    # Fit Camera Parameters 
    trace = cam.metropolis([
            ct.FitParameter("elevation_m", lower=0, upper=100, value=20),
            ct.FitParameter("tilt_deg", lower=0, upper=180, value=80),
            ct.FitParameter("heading_deg", lower=-180, upper=180, value=-77),
            ct.FitParameter("roll_deg", lower=-180, upper=180, value=0)
            ], iterations=1e4)
finally:
    pass
