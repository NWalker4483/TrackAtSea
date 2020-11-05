import os 
import cv2
def rip_frames(vid_path, out_path=None):
    '''
    Brief:
        Parses a video file and saves each frame as a png.

    Parameters:
        vid_path (string) Path to video file.
        out_path (string) Path where frame png file will be saved.

    Returns:
        None
    '''

    if out_path == None:
        print("Please specify a directory to save frames")
    else:
        if os.path.exists(out_path):
            vidcap = cv2.VideoCapture(vid_path)
            success,image = vidcap.read()
            count = 0
            print("Ripping frames...")
            while success:
                if (count % 100) == 0:
                    path = os.path.join(out_path, f"{vid_path.split('/')[-1].split('.')[0]}.{count}.png")
                    cv2.imwrite(path, image)
                success,image = vidcap.read()
                count += 1
            print("Saved {} frames to {}.".format(count, out_path))
        else:
            print("Specified directory does not exist")

if __name__ == "__main__":
    for i in range(6,14):
        rip_frames(f"raw_data/video/{i}.mp4","generated_data/images/")