'''
A Naïve approach to assigning a GPS value to every frame of each video by 
subtracting the time estimates for each frame and finding the 
GPS reading with the minimum absolute difference 

Associating GPS readings with a frame of a video
'''
import xml.etree.ElementTree as ET
import csv
import tqdm
import numpy as np

def time2secs(time):  # Convert a string representation of a H:M:S into the number of seconds
    Hours, Min, Sec = [int(i) for i in time.split(":")]
    return 360 * Hours + 60 * Min + Sec


tree = ET.parse(
    'raw_data/main_boat_position/onboard_gps_source1/AI Tracks at Sea High Frequency GPS_train.txt')
root = tree.getroot()
readings = list()
for entry in root.iter("trkpt"):
    readings.append((entry.find("time").text.split("T")[1][:-1],\
                  float(entry.attrib["lat"]), float(entry.attrib["lon"]))) # (Time in seconds, latitude, longitude)

frame_times = dict()
def dist(p1,p2):
    return (((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)) ** .5

for i in range(6, 23):
    frame_times[i] = dict()
    with open(f"raw_data/camera_gps_logs/SOURCE_GPS_LOG_{i}_cleaned.csv","r") as f:
       content = f.readlines()
       content = content[1:] # Skip Header
    for entry in content:  
        frame_id, utc_time, _, _, est_time = entry.split(",")
        frame_times[i][frame_id] = utc_time.strip()
try:
    files = dict()
    [files.update({i : open(f"generated_data/frame2gps/{i}.csv", "w+")}) for i in frame_times]
    writers = dict()
    for _file in files:
        fields = ['Frame No.','Frame Time','GPS Time', 'Latitude', 'Longitude']  
        csvwriter = csv.writer(files[_file])  
        csvwriter.writerow(fields)  
        writers[_file] = csvwriter
    results = dict()
    # * Could I optimize this... definitely. Is it worth the mental effort probably not

    for video in tqdm.tqdm(frame_times):
        last_best = None
        results[video] = []
        for frame in frame_times[video]:
            best_reading = None
            best_score = np.inf
            for reading in readings:
                score = abs(time2secs(frame_times[video][frame]) - time2secs(reading[0]))
                if (score < best_score) and (score <= 5):
                    best_reading = reading 
                    best_score = score
            if best_reading != None:
                results[video].append([frame, frame_times[video][frame], best_reading[0], best_reading[1], best_reading[2]])
    
    ##############################################
    # ? I havent evalutated the helpulness of this fully And some of the assumptions aren't necessarily true
    valid_results = dict()
    for video in tqdm.tqdm(results):
        prev_gps = (np.inf, np.inf)
        groups = [] #### Group Frames by the assigned gps reading
        for frame in results[video]:
            gps = (frame[3], frame[4])
            if prev_gps == gps:
                groups[-1].append(frame)
            else:
                groups.append([frame])
            prev_gps = gps
        
        while True:
            perfect = True 
            temp = []
            for i in range(1,len(groups)-1):
                
                prev_pnt = [groups[i-1][0][3],groups[i-1][0][4]]
                this_pnt = [groups[i][0][3],    groups[i][0][4]]
                next_pnt = [groups[i+1][0][3],groups[i+1][0][4]]
            
                # Remove Sharp Changes in gps 
                if dist(prev_pnt, this_pnt) <= dist(prev_pnt, next_pnt):
                    temp.append(groups[i])
                else:
                    perfect = False
            groups = temp 
            if perfect: break 
        for group in groups:
            for row in group:
                writers[video].writerow(row)
finally:  
    for _file in files:
        files[_file].close()