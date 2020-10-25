import argparse
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_file', type=str,
                    help='Video file name (MP4 format)')
parser.add_argument('-n', type=int,
                    help='Number of points to be generated')
parser.add_argument('--cam_lat', type=float,
                    help='Latitude of source in decimal degrees (i.e. where the camera is mounted))
parser.add_argument('--cam_long', type=float,
                    help='Longitude of source in decimal degrees (i.e. where the camera is mounted))
parser.add_argument('--tracker_type', type=str, default="Manual"
                    help='')
parser.add_argument('--cam_transform_path', type=str, default="data/transforms/manual"
                    help='')

args = parser.parse_args()

video = cv2.VideoCapture(args.video_file)
if args.tracker_type == "MVDA":
    from MVDA import MVDATracker
    tracker = MVDATracker(init_frames=50, detecting_rate=1, detections_per_denoising=5, framerate=20, max_recovery_distance= 50 )

elif args.tracker_type == "DEEP":
    pass
try:
    while cap.isOpened():
        ret, frame = video.read()
finally:
    pass
