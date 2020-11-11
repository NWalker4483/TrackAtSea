'''
A Naïve approach to assigning a GPS value to every frame of each video by 
subtracting the time estimates for each frame and finding the 
GPS reading with the minimum absolute difference 

Associating GPS readings with a frame of a video
'''
import xml.etree.ElementTree as ET
import csv

def time2secs(time):  # Convert a string representation of a H:M:S into the number of seconds
    Hours, Min, Sec = [int(i) for i in time.split(":")]
    return 360 * Hours + 60 * Min + Sec


tree = ET.parse(
    'raw_data/main_boat_position/onboard_gps_source1/AI Tracks at Sea High Frequency GPS_train.txt')
root = tree.getroot()
readings = list()
for entry in root.iter("trkpt"):
    readings.append((time2secs(entry.find("time").text.split("T")[1][:-1]),\
                  float(entry.attrib["lat"]), float(entry.attrib["lon"]))) # (Time in seconds, latitude, longitude)

frame_times = dict()
for i in range(6, 23):
    frame_times[i] = dict()
    with open(f"raw_data/camera_gps_logs/SOURCE_GPS_LOG_{i}_cleaned.csv","r") as f:
       content = f.readlines()
       content = content[1:] # Skip Header
    for entry in content:  
        frame_id, utc_time, _, _, est_time = entry.split(",")
        frame_times[i][frame_id] = time2secs(utc_time)

try:
    files = dict()
    [files.update({i : open(f"generated_data/gps2frame/gps2frame.{i}.csv", "w")}) for i in frame_times]
    writers = dict()
    for _file in files:
        fields = ['Frame No.', 'Latitude', 'Longitude']  
        csvwriter = csv.writer(files[_file])  
        csvwriter.writerow(fields)  
        writers[_file] = csvwriter
    # * Could I optimize this... definitely. Is it worth the mental effort probably not
    for reading in readings:
        best_err = 999999999
        best_video = -1
        best_frame = -1  
        
        for video in frame_times:
            for frame in frame_times[video]:
                score = abs(frame_times[video][frame] - reading[0])
                if (score < best_err):
                    best_video = video 
                    best_frame = frame 
                    best_err = score
        writers[best_video].writerow([best_frame, reading[1], reading[2]])
finally:  
    for _file in files:
        files[_file].close()