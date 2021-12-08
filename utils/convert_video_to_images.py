import cv2
import os, sys
from tqdm import tqdm
import math

def convert(videos_dir, fps, file=None):

    if file != None: 
        videos = list(open(file, 'r'))

    else: videos = [x for x in os.listdir(videos_dir) if x.endswith(".avi")]

    for video in videos:

        videos_dir = videos_dir.replace('\\','/')
        file_dir = videos_dir + '/' + video
        save_dir = file_dir.split('.')[0]

        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
            
            vidcap = cv2.VideoCapture(file_dir)
            vidlength = math.floor(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vidfps = math.ceil(vidcap.get(cv2.CAP_PROP_FPS))
            print(video)
            print('frames: {}, fps: {}'.format(vidlength, vidfps))
            if int(fps) > vidfps: fps = vidfps
            frames = vidlength / (vidfps / int(fps))
            print('number: {:.0f}'.format(frames))
            
            #if vidlength / (vidfps * 60) > 10:  # over 10 mins
            #    continue
            count = 0
            for i in tqdm(range(vidlength)):
                
                success, image = vidcap.read()
                if not success: print('frame {} broken'.format(i))

                if (i + 1) % (vidfps / int(fps)) == 0:

                    framecount = "{number:06}".format(number=count)
                    filename = save_dir + '/' + framecount + ".jpg"

                    if not os.path.exists(filename):
                        cv2.imwrite(filename, image)     # save frame as JPEG file      
                        
                    if not success: print('missing {}.jpg'.format(framecount))

                    count += 1
        

if __name__ == "__main__":
    if len(sys.argv) < 3:
        "Please give a video as an argument to this program"
    elif len(sys.argv) == 3:
        convert(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        convert(sys.argv[1], sys.argv[2], sys.argv[3])
