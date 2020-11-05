import argparse
import cameratransform as ct
import pickle
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_num', type=int, default=None)

parser.add_argument('--video_file', type=str, default="raw_data/video/6.mp4",
                    help='Video file name (MP4 format)')
parser.add_argument('--gps_list_file', type=str, default="generated_data/frame2gps/frame2gps.6.list",
                    help='')

parser.add_argument('--num_detections', type=int, default=10,
                    help='')
parser.add_argument('--cam_lat', type=float, default = 32.70297, 
                    help='Latitude of source in decimal degrees (i.e. where the camera is mounted')
parser.add_argument('--cam_long', type=float, default = -117.23463100000001,
                    help='Longitude of source in decimal degrees (i.e. where the camera is mounted')
parser.add_argument('--tracker_type', type=str, default="Manual",
                    help='')

args = parser.parse_args()

if args.video_num != None:
    args.video_file = f"raw_data/video/{args.video_num}.mp4"
    args.gps_list_file = f"generated_data/frame2gps/frame2gps.{args.video_num}.list"

video = cv2.VideoCapture(args.video_file)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frames_read = 0 

# intrinsic camera parameters
f = 6.2    # in mm
sensor_size = (6.17, 4.55)    # in mm
image_size = (720, 1280)    # in px

# initialize the camera
cam = ct.Camera(ct.RectilinearProjection(focallength_mm=f,
                                         sensor=sensor_size,
                                         image=image_size)) #,lens = ct.BrownLensDistortion()))
cam.setGPSpos(args.cam_lat, args.cam_long)

with open(args.gps_list_file, "rb") as fp:   # Unpickling
    lm_points_gps = np.array(pickle.load(fp))
    lm_points_gps = np.hstack((lm_points_gps, np.ones((lm_points_gps.shape[0],1)) * 15))
    lm_points_space = cam.spaceFromGPS(lm_points_gps)
assert(len(lm_points_space) == frame_count)

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
        cam.addLandmarkInformation(np.array(px), space_coords, [3, 3, 10])
    # Fit Camera Parameters
    print("Fitting Transform")
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    trace = cam.metropolis([
            ct.FitParameter("elevation_m", lower=0, upper=100, value=10),
            ct.FitParameter("tilt_deg", lower=0, upper=180, value=0),
            ct.FitParameter("heading_deg", lower=-180, upper=180, value=0),
            ct.FitParameter("roll_deg", lower=-180, upper=180, value=0)
            ], iterations=10000)
    # cam.plotTrace()
    # plt.tight_layout()
finally:
    cam.save(f"generated_data/transforms/{args.tracker_type.lower()}_transform.json") # ? maybe add a time to the filename