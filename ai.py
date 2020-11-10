from imageai.Detection.Custom import CustomObjectDetection


import os
import cv2

execution_path = os.getcwd()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "generated_data/target_ship/models/detection_model.h5"))
detector.setJsonPath(os.path.join(execution_path, "generated_data/target_ship/json/detection_config.json"))
detector.loadModel()

cap = cv2.VideoCapture(os.path.join(execution_path, "raw_data/video", "6.mp4"))
output_video = cv2.VideoWriter(os.path.join(execution_path, "video-result.avi"), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                24, ((int(cap.get(3)), int(cap.get(4)))))

progress_tracker = 0
response_label = ""
skip_frame = 5

while (cap.isOpened()):
    valid, frame = cap.read()

    if valid == True:
        progress_tracker += 1

        if (progress_tracker % skip_frame == 0):

            cv2.imwrite("video_image.jpg", frame)

            try:
                detections = detector.detectObjectsFromImage(input_image="video_image.jpg", output_image_path="holo1-detected.jpg")
                for detection in detections:
                    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
                    x1, y1, x2, y2 = detection["box_points"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except:
                None

        font = cv2.FONT_HERSHEY_PLAIN
        frame = cv2.putText(frame, '{}'.format(response_label), (150, 35), font, 3, (207, 109, 4), 3, cv2.LINE_AA)
        cv2.imshow('Image Viewer', frame)

        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
