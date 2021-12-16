import os
from PIL import Image
import numpy as np

class dataset():

    def __init__(self, root, movie_file, batch_size, frames_size, crop_size, 
                crop_mean=False, rescale=False):
        
        self.dataset = []
        self.movie_name = []
        self.root = root
        self.movie_file = movie_file
        self.batch_size = batch_size
        self.frames_size = frames_size
        self.crop_size = crop_size
        self.crop_mean = crop_mean
        self.rescale = rescale
 
        with open(movie_file, 'r') as f:
            for line in f:
                video_name = line.split('-')[0] + '-' + line.split('-')[1]
                if line[0] != '_': self.movie_name.append(video_name)
        
        print('movie number: {}'.format(len(self.movie_name)))

    def __len__(self):

        return len(self.movie_name)
    
    def load_frames(self, vid, start):

        images = [x for x in os.listdir(os.path.join(self.root, vid)) if x.endswith(".jpg")]

        frames = []
        index = 0
        for i in range(start * self.frames_size, (start + 1) * self.frames_size):

            if i > len(images): return [], 0

            img = Image.open(os.path.join(self.root, vid, images[index]))

            if img.width > img.height:
                scale = float(self.crop_size) / float(img.height)
                img = np.array(img.resize([int(img.width * scale + 1), self.crop_size], Image.ANTIALIAS))
            else:
                scale = float(self.crop_size) / float(img.width)
                img = np.array(img.resize([self.crop_size, int(img.height * scale + 1)], Image.ANTIALIAS))

            crop_x = int((img.shape[0] - self.crop_size) / 2) # center 
            crop_y = int((img.shape[1] - self.crop_size) / 2) 

            img = img[crop_x : crop_x + self.crop_size, crop_y : crop_y + self.crop_size, :]

            frames.append(img)

        return frames, start + 1

    def sample_frames(self, vid, start, sample, overlap=0.8):

        images = [x for x in os.listdir(os.path.join(self.root, vid).replace('\\','/')) if x.endswith(".jpg")]
        #print('number: {}'.format(len(images)))

        if self.crop_mean: 
            np_mean = np.load('./extractor/crop_mean.npy').reshape(
                                [self.frames_size, self.crop_size, self.crop_size, 3])

        frames = []
        count = 0
        index = int(start * sample * (1 - overlap))

        if index + sample >= len(images): return [], -1

        for i in range(index, index + sample):

            if i % (sample / self.frames_size) == 0:

                img = Image.open(os.path.join(self.root, vid, images[i]))

                if img.width > img.height:
                    scale = float(self.crop_size) / float(img.height)
                    img = np.array(img.resize([int(img.width * scale + 1), self.crop_size], Image.ANTIALIAS))
                else:
                    scale = float(self.crop_size) / float(img.width)
                    img = np.array(img.resize([self.crop_size, int(img.height * scale + 1)], Image.ANTIALIAS))

                # center crop
                crop_x = int((img.shape[0] - self.crop_size) / 2) 
                crop_y = int((img.shape[1] - self.crop_size) / 2) 

                img = img[crop_x : crop_x + self.crop_size, crop_y : crop_y + self.crop_size, :].astype(np.float32)
                
                if self.crop_mean: 
                    img -= np_mean[count]
                    count += 1

                elif self.rescale:
                    img = (img / 255.) * 2 - 1

                frames.append(img)

        return frames, start + 1

    def next_batch(self, start_pos, batch_index, sample, overlap=0.8):
        
        movie_name = self.movie_name[start_pos]
        next_start = start_pos
        next = batch_index

        batch = []
        count = 0
        while count < self.batch_size:

            if next_start >= len(self.movie_name): 
                return np.array(batch), None, len(self.movie_name), -1
            
            movie_name = self.movie_name[next_start]
            images, next = self.sample_frames(movie_name, next, sample, overlap)

            if len(images) != 0: 
                batch.append(images)
                count += 1

            if next == -1: 
                next = 0
                next_start += 1
                if len(batch) != 0: break

        if len(batch) < self.batch_size:

            temp = batch[len(batch) - 1]
            for i in range(self.batch_size - len(batch)):
                batch.append(temp)

        return np.array(batch, dtype=np.float32), movie_name, next_start, next
