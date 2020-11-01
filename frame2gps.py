'''
A Naïve approach to assigning a GPS value to every frame of each video by 
subtracting the time estimates for each frame and finding the 
GPS reading with the minimum absolute difference 
'''
import xml.etree.ElementTree as ET
import pickle


def time2secs(time):  # Convert a string representation of a H:M:S into the number of seconds
    Hours, Min, Sec = [int(i) for i in time.split(":")]
    return 360 * Hours + 60 * Min + Sec


tree = ET.parse(
    'raw_data/main_boat_position/onboard_gps_source1/AI Tracks at Sea High Frequency GPS_train.txt')
root = tree.getroot()
times = list()
for entry in root.iter("trkpt"):
    times.append((entry.find("time").text.split("T")[
                 1][:-1], entry.attrib["lat"], entry.attrib["lon"]))

for i in range(6, 23):
    video_info_file = f"raw_data/camera_gps_logs/SOURCE_GPS_LOG_{i}_cleaned.csv"
    gps_est = []
    with open(video_info_file, "r") as f:
        content = f.readlines()
    for entry in content[1:]:  # Skip Header
        frame_id, utc_time, lat, lon, est_time = entry.split(",")
        # * Could be way faster with binary search but it only runs once so ehh its not a big deal
        best_est = min(times, key=lambda x: abs(
            time2secs(x[0])-time2secs(utc_time)))
        gps_est.append([float(coord) for coord in best_est[1:]])
    # Pickling
    with open(f"generated_data/frame2gps/frame2gps.{i}.list", "wb") as fp:
        pickle.dump(gps_est, fp)
