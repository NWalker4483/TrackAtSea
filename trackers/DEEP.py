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
        detections = self.detector.detectObjectsFromImage(input_image="temp.jpg", output_image_path="temp2.jpg")
        updates = [(self.frames_read, 0, detection["box_points"]) for detection in detections]
        self.frames_read += 1
        return updates
  

if __name__ == "__main__":

    execution_path = os.getcwd()
    cap = cv2.VideoCapture(os.path.join(execution_path, "raw_data/video", f"{6}.mp4"))
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
