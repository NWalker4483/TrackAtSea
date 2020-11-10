
# ! And I believe that the below assumption is incorrect from testing
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


# https://stackoverflow.com/questions/19079619/efficient-way-to-combine-intersecting-bounding-rectangles
def combineBoundingBox(box1, box2):
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = box2[0] + box2[2] - box1[0]
    h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
    return (x, y, w, h)

# * Can be simplified from the code sample


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
    return img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
