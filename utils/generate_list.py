import os
import sys
import random

SAVE_DIRECTORY = 'E:/File/VS Code/TensorFlow/TAL/dataset'
IMAGES_DIRECTORY = 'E:/File/VS Code/DataSet/TACoS/MPII-Cooking-2-videos'

if not os.path.exists(IMAGES_DIRECTORY):

    print(IMAGES_DIRECTORY + " not exist")
    exit()

else: print(IMAGES_DIRECTORY)

trainval_percent = 0.7 # train + val
val_percent = 0.0 # val in train

video_files = []
#os.chdir(os.path.join("data", "obj"))
for filename in os.listdir(IMAGES_DIRECTORY):
#for root, dirs, files in os.walk(IMAGES_DIRECTORY):
    #print(files)
    video_files.append(IMAGES_DIRECTORY + '/' + filename)
#video_files.remove(IMAGES_DIRECTORY)

print('num of video: ' + str(len(video_files)))

num = len(video_files)
tv = int(num * trainval_percent)
tr = int(tv * val_percent)
random.shuffle(video_files)
train_set = video_files[:tv]
val_set = random.sample(train_set, tr)

with open(SAVE_DIRECTORY + "/train.txt", "w") as outfile:
    for video in train_set:
        outfile.write(video + '\n')

with open(SAVE_DIRECTORY + "/val.txt", "w") as outfile:
    for video in val_set:
        outfile.write(video  + '\n')
    
if num > tv:
    test_set = video_files[tv:]
    with open(SAVE_DIRECTORY + "/test.txt", "w") as outfile:
        for video in test_set:
            outfile.write(video + '\n')
        
print("complete")
