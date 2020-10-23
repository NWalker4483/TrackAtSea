import numpy as np
import cv2
from MVDA import MVDATracker

cap = cv2.VideoCapture('data/video/6.mp4')
def crop_bottom_half(image):
    cropped_img = image[int(image.shape[0]/2):image.shape[0]]
    return cropped_img
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
smp = crop_bottom_half(cap.read()[1])
out = cv2.VideoWriter('output.mp4',fourcc, 15, (smp.shape[1],  smp.shape[0]*3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
try:
    while(1):
        ret, frame = cap.read()
        frame = crop_bottom_half(frame)
        temp = frame 

        mask = fgbg.apply(temp)
        res = cv2.bitwise_and(frame,frame,mask = mask)
        max_area_only = True 
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if max_area_only and len(contours) > 0:
            c = max(contours, key=lambda x: cv2.contourArea(x))
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(frame,'Boat Detected ' + str(cv2.contourArea(c)) ,(x+w+10,y+h),0,0.3,(0,255,0))

        else:
            for c in [cnt for cnt in contours if cv2.contourArea(cnt) > 75]:
                rect = cv2.boundingRect(c)
                x,y,w,h = rect
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                cv2.putText(frame,'Boat Detected ' + str(cv2.contourArea(c)) ,(x+w+10,y+h),0,0.3,(0,255,0))
        view = cv2.vconcat([frame,res,cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)])
        out.write(view)
        cv2.imshow('frame',view)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()