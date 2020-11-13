# Trained on Google colab
from imageai.Detection.Custom import CustomObjectDetection

import os
import cv2
class DEEPTracker():
    def __init__(self, skip_frames = 20, horizon = 420):
        execution_path = os.getcwd()
        detector = CustomObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(execution_path, f"generated_data/target_ship/models/detection_model.h5"))
        detector.setJsonPath(os.path.join(execution_path, f"generated_data/target_ship/json/detection_config.json"))
        detector.loadModel()

        self.horizon = horizon
        self.detector = detector
        self.frames_read = 0
    def update(self, frame):
        cv2.imwrite("temp.jpg", frame)
        detections = self.detector.detectObjectsFromImage(input_image="temp.jpg", output_image_path="temp.2.jpg")
        updates = [(self.frames_read, 0, detection["box_points"]) for detection in detections]
        self.frames_read += 1
        return updates
  

if __name__ == "__main__":
    import argparse
    import random
    import pickle 
    import csv 

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video_num', type=int, required=True)

    parser.add_argument('--video_file', type=str, default="raw_data/video/7.mp4",
                        help='Video file name (MP4 format)')
    
    parser.add_argument('--num_detections', type=int, default=20,
                        help='')

    args = parser.parse_args()

    if args.video_num != None:
        args.video_file = f"raw_data/video/{args.video_num}.mp4"

    cap = cv2.VideoCapture(args.video_file)

    execution_path = os.getcwd()
    output_video = cv2.VideoWriter(os.path.join(execution_path, "output.deep.avi"), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                    24, ((int(cap.get(3)), int(cap.get(4)))))

    response_label = "target"
    tracker = DEEPTracker()

    while (cap.isOpened()):
        valid, frame = cap.read()

        if not valid: break

        detections = tracker.update(frame)
        for frame_num, ID, box in detections:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # font = cv2.FONT_HERSHEY_PLAIN
        # frame = cv2.putText(frame, '{}'.format(response_label), (150, 35), font, 3, (207, 109, 4), 3, cv2.LINE_AA)
        cv2.imshow('Image Viewer', frame)

        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
