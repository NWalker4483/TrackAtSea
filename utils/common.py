import cv2
import numpy as np 
cache = {}
def id_to_random_color(number):
    if not number in cache:
        r, g, b = np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)
        cache[number]= (r, g, b)
        return r, g, b
    else:
        return cache[number]


def remove_contained(boxes):  # ? The paper was not clear whether or not partially overlapping or entirely overlapping boxes should be removed as of now only fully overlapping boxes will be
    contained = set()
    final = set()
    for container in boxes:
        if container not in contained:
            for box in boxes:

                if (container[0] <= box[0]) and (container[1] <= box[1]):  # Higher
                    # Longer and Taller
                    if (container[0] + container[2] >= box[0] + box[2]) and (container[1] + container[3] >= box[1] + box[3]):
                        if box != container:
                            contained.add(box)
    return set([box for box in boxes if box not in contained])


def merge_boxes(boxes):
    seen = set()
    new_boxes = set()
    for boxA in boxes:
        seen.add(boxA)
        merged = False
        for boxB in boxes:
            if boxB not in seen:  # Only Check only once and dont check against self
                if intersection_over_union(boxA, boxB) > 0:  # Touching
                    new_boxes.add(combineBoundingBox(boxA, boxB))
                    merged = True
        if not merged:
            new_boxes.add(boxA)
    return new_boxes if new_boxes == boxes else merge_boxes(new_boxes)


def combineBoundingBox(box1, box2): # https://stackoverflow.com/questions/19079619/efficient-way-to-combine-intersecting-bounding-rectangles
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = box2[0] + box2[2] - box1[0]
    h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
    return (x, y, w, h)


def intersection_over_union(boxA, boxB):
    boxA = list(boxA)
    boxB = list(boxB)
    # Convert from cv2 to (xy,x,y)
    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def interesection_area(boxA, boxB):
    boxA = list(boxA)
    boxB = list(boxB)
    # Convert from cv2 to (xy,x,y)
    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea


def distance_between_centers(boxA, boxB):
    x1, y1 = boxA[0] + (boxA[2] // 2), boxA[1] + (boxA[3] // 2)
    x2, y2 = boxB[0] + (boxB[2] // 2), boxB[1] + (boxB[3] // 2)
    return (((x1-x2)**2)+((y1-y2)**2)) ** .5


def crop_to(img, box):
    return img


def load(video_num,preface = "manual", train = False):
    det_frames = dict()
    with open(f"generated_data/tracks/{preface}.{video_num}.csv","r") as f:
        content = f.readlines()
        content = content[1:] # Skip Header
    for entry in content:
        Frame_No, ID, X1, Y1, X2, Y2 = [int(i) for i in entry.split(",")]
        if train and int(ID) != -1: continue
        det_frames[int(Frame_No)] = [X1, Y1, X2, Y2]

    gps_points = []
    distorted_points = []
    with open(f"generated_data/frame2gps/{video_num}.csv","r") as f:
        content = f.readlines()
        content = content[1:] # Skip Header
    for entry in content:
        Frame_No, Frame_Time, GPS_Time, Latitude, Longitude = entry.split(",")
        if int(Frame_No) in det_frames:
            gps_points.append([float(Latitude), float(Longitude)])
            distorted_points.append(det_frames[int(Frame_No)])
    return np.array(distorted_points, dtype=np.float64), np.array(gps_points, dtype=np.float64)
def load_many(video_nums, preface = "manual", train = False):
    distorted_points, gps_points = load(video_nums[0])
    for i in range(1,len(video_nums)):
        distorted_points_temp, gps_points_temp = load(video_nums[i], preface, train)
        distorted_points = np.concatenate((distorted_points, distorted_points_temp), axis=0)
        gps_points = np.concatenate((gps_points, gps_points_temp), axis=0)
    return distorted_points, gps_points 


def plot_gps(data, height = 7000, buffer = 1000, base_img=None,colors = None): # [[Lat, Long],[Lat, Long], ...]
    # Normalize
    min_lat = np.inf
    max_lat = -np.inf
    min_lon = np.inf
    max_lon = -np.inf
    for path in data: 
        for lat, lon in path:
            min_lat = lat if lat < min_lat else min_lat
            max_lat = lat if lat > max_lat else max_lat
            min_lon = lon if lon < min_lon else min_lon
            max_lon = lon if lon > max_lon else max_lon
    #print(min_lat,max_lat,min_lon,max_lon)
    
    change_per_px = height / (max_lat - min_lat) # Find the largest change in gps 
    if base_img == None:
        base_img = np.ones((buffer + int(change_per_px * (max_lon - min_lon)),buffer + height ,3)) * 255
    if colors == None:
        colors = [tuple([np.random.randint(0,255) for _ in range(3)]) for i in range(len(data))]
    legend_y = int(50 * (height/1000))
    i = 1
    for path in data:
        path = np.array(path)
        color = colors[i-1]
        base_img = cv2.putText(base_img, f'Path {i}', (50, legend_y*i), cv2.FONT_HERSHEY_SIMPLEX ,  
                           height/1000, color, 7, cv2.LINE_AA) 
        i+=1
        #color = (255,23,0)
        path[:,0] -= min_lat
        path[:,1] -= min_lon
        last_point = ((buffer//4) + int(path[0][0] * change_per_px), (buffer//4) + int(path[0][1] * change_per_px))
        for d_lat, d_lon in path[1:]:
            point = ((buffer//4) + int(d_lat * change_per_px), (buffer//4) + int(d_lon * change_per_px))
            base_img = cv2.circle(base_img, point,10, color, -1) 
            base_img = cv2.line(base_img, last_point, point, color, 10) 
            last_point = point
    cv2.imwrite("gps.png",base_img)