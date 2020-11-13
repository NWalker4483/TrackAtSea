import cv2

det_frames = dict()
with open(f"generated_data/tracks/manual.{args.video_num}.csv","r") as f:
    content = f.readlines()
    content = content[1:] # Skip Header
for entry in content:
    Frame_No, ID, X1, Y1, X2, Y2 = [int(i) for i in entry.split(",")]
    det_frames[int(Frame_No)] = [X1+((X2 - X1)//2),Y2]