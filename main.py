import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_file', type=str,
                    help='Video file name (MP4 format)')
parser.add_argument('-n', type=int,
                    help='Number of points to be generated')
parser.add_argument('--cam_lat', type=float,
                    help='Latitude of source in decimal degrees (i.e. where the camera is mounted))
parser.add_argument('--cam_long', type=float,
                    help='Longitude of source in decimal degrees (i.e. where the camera is mounted))
parser.add_argument('--detector_type', type=str, default="MVDA"
                    help='')
parser.add_argument('--cam_param_path', type=str, default=""
                    help='')

args = parser.parse_args()