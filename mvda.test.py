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
fgbg = MVDATracker(init_frames=50, detecting_rate=1, detections_per_denoising=5, framerate=20, max_recovery_distance= 50 )
try:
    while(1):
        ret, frame = cap.read()
        frame = crop_bottom_half(frame)
        rects = fgbg.update(frame)
        mask = fgbg.background_mask
        res = cv2.bitwise_and(frame,frame,mask = mask)
        for rect in rects:
            ID, rect = rect
            x,y,w,h = rect
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
            # cv2.putText(frame,f' {ID} ',(x+w+10,y+h),0,0.3,(0,255,255))
        for rect in fgbg.last_states:
            ID, rect = rect
            x,y,w,h = rect
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,f'Boat {ID} Detected ',(x+w+10,y+h),0,0.3,(0,255,0))
    
        view = cv2.vconcat([frame,res,cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)])
        out.write(view)
        cv2.imshow('frame',view)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()