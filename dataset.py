
import numpy as np
import os
import random
import pickle

'''
calculate temporal intersection over union
'''
def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1] - inter[0]
    length = sliding_clip[1] - sliding_clip[0]
    nIoL = 1.0 * (length - inter_l) / length
    return nIoL

class TrainingDataSet(object):
    def __init__(self, sliding_dir, it_path, batch_size):
        
        self.sliding_clip_path = sliding_dir
        self.batch_size = batch_size
        self.context_num = 1
        self.context_size = 128
        self.visual_feature_dim = 1024
        self.sent_vec_dim = 4800

        movie_names_set = set()
        print ("Reading training data list from " + it_path)
        csv = pickle.load(open(it_path, 'rb'), encoding="latin1")
        self.clip_sentence_pairs = []
        for k, v in csv.items():
            clip_name = k
            sent_vecs = v
            movie_name = clip_name.split("_")[0]
            if not clip_name in movie_names_set:
                movie_names_set.add(clip_name)
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))
        print(str(len(self.clip_sentence_pairs)) + " clip-sentence pairs are readed")

        self.movie_names = list(movie_names_set)
        self.num_samples = len(self.clip_sentence_pairs)
        # read sliding windows, and match them with the groundtruths to make training samples
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.clip_sentence_pairs_iou = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[1] == "npy": # ex: s13-d21_0_64.npy
                movie_name = clip_name.split("_")[0] # ex: s13-d21
                for clip_sentence in self.clip_sentence_pairs:
                    sentence_clip_name = clip_sentence[0]  # ex: s13-d21_252_452
                    sentence_movie_name = sentence_clip_name.split("_")[0] # ex: s13-d21
                    if sentence_movie_name == movie_name:
                        start = int(clip_name.split("_")[1])
                        end = int(clip_name.split("_")[2].split(".")[0])
                        o_start = int(sentence_clip_name.split("_")[1]) 
                        o_end = int(sentence_clip_name.split("_")[2])
                        iou = calculate_IoU((start, end), (o_start, o_end))  
                        if iou > 0.5:
                            nIoL = calculate_nIoL((o_start, o_end), (start, end))
                            if nIoL < 0.15:
                                start_offset = o_start - start
                                end_offset = o_end - end
                                self.clip_sentence_pairs_iou.append((clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))
        
        self.num_samples_iou = len(self.clip_sentence_pairs_iou)
        print(str(len(self.clip_sentence_pairs_iou)) + " iou clip-sentence pairs are readed")
       
    '''
    compute left (pre) and right (post) context features
    '''
    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        clip_length = self.context_size
        left_context_feats = np.zeros([win_length, self.visual_feature_dim], dtype=np.float32)
        right_context_feats = np.zeros([win_length, self.visual_feature_dim], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+clip_name)

        for k in range(win_length):
            left_context_start = start - clip_length * (k + 1)
            left_context_end = start - clip_length * k
            right_context_start = end + clip_length * k
            right_context_end = end + clip_length * (k + 1)
            left_context_name = '{}_{}_{}.npy'.format(movie_name, left_context_start, left_context_end)
            right_context_name = '{}_{}_{}.npy'.format(movie_name, right_context_start, right_context_end)

            if os.path.exists(self.sliding_clip_path + left_context_name):
                left_context_feat = np.load(self.sliding_clip_path + left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat

            if os.path.exists(self.sliding_clip_path + right_context_name):
                right_context_feat = np.load(self.sliding_clip_path + right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat

            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return left_context_feats, right_context_feats
    
    '''
    read next batch of training data, this function is used for training CTRL-aln
    '''
    def next_batch(self):
        
        random_batch_index = random.sample(range(self.num_samples), self.batch_size)
        image_batch = np.zeros([self.batch_size, self.visual_feature_dim])
        sentence_batch = np.zeros([self.batch_size, self.sent_vec_dim])
        offset_batch = np.zeros([self.batch_size, 2], dtype=np.float32) # this one is actually useless

        index = 0
        clip_set=set()
        while index < self.batch_size:
            k = random_batch_index[index]
            clip_name = self.clip_sentence_pairs[k][0]
            if not clip_name in clip_set: 
                clip_set.add(clip_name)
                feat_path = self.sliding_clip_path + self.clip_sentence_pairs[k][0]
                featmap = np.load(feat_path)
                image_batch[index,:] = featmap
                sentence_batch[index,:] = self.clip_sentence_pairs[k][1][:self.sent_vec_dim]
                index+=1
            else:
                r = random.choice(range(self.num_samples))
                random_batch_index[index] = r
                continue 
                      
        return image_batch, sentence_batch, offset_batch

    '''
    read next batch of training data, this function is used for training CTRL-reg
    '''
    def next_batch_iou(self):

        random_batch_index = random.sample(range(self.num_samples_iou), self.batch_size)
        image_batch = np.zeros([self.batch_size, self.visual_feature_dim, 2 * self.context_num + 1])
        sentence_batch = np.zeros([self.batch_size, self.sent_vec_dim])
        offset_batch = np.zeros([self.batch_size, 2], dtype=np.float32)

        index = 0
        clip_set = set()
        while index < self.batch_size:
            k = random_batch_index[index]
            sentence_clip_name = self.clip_sentence_pairs_iou[k][0]
            if not sentence_clip_name in clip_set:
                clip_set.add(sentence_clip_name)
                feat_path = self.sliding_clip_path + self.clip_sentence_pairs_iou[k][2]   # clip_name
                featmap = np.load(feat_path)
                # read context features
                left_context_feat, right_context_feat = self.get_context_window(self.clip_sentence_pairs_iou[k][2], self.context_num)
                left_context_feat = np.transpose(left_context_feat, [1,0])
                right_context_feat = np.transpose(right_context_feat, [1,0])
                image_batch[index,:,:] = np.column_stack((left_context_feat, featmap, right_context_feat))
                # image_batch[index,:] = np.hstack((left_context_feat, featmap, right_context_feat))
                sentence_batch[index,:] = self.clip_sentence_pairs_iou[k][1][:self.sent_vec_dim]
                p_offset = self.clip_sentence_pairs_iou[k][3]   # start_offset
                l_offset = self.clip_sentence_pairs_iou[k][4]   # end_offset
                offset_batch[index,0] = p_offset
                offset_batch[index,1] = l_offset
                index += 1
            else:
                r = random.choice(range(self.num_samples_iou))
                random_batch_index[index] = r
                continue
        
        return image_batch, sentence_batch, offset_batch

class TestingDataSet(object):
    def __init__(self, img_dir, csv_path, batch_size):

        self.sliding_clip_path = img_dir
        self.batch_size = batch_size
        self.visual_feature_dim = 1024
        self.semantic_size = 4800

        movie_names_set = set()
        print("Reading testing data list from " + csv_path)
        csv = pickle.load(open(csv_path, 'rb'), encoding="latin1")
        self.clip_sentence_pairs = []
        for k, v in csv.items():
            clip_name = k
            sent_vecs = v
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))

        print(str(len(self.clip_sentence_pairs)) + " pairs are readed")
        
        self.movie_names = list(movie_names_set)
        self.num_samples = len(self.clip_sentence_pairs)
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.sliding_clip_names = []
        for clip_name in sliding_clips_tmp: # ex: s13-d21_1_65.npy
            if clip_name.split(".")[1] == "npy":
                movie_name = clip_name.split("_")[0] # s13-d21
                if movie_name in self.movie_names:
                    self.sliding_clip_names.append(clip_name)    

        print("sliding clips number: " + str(len(self.sliding_clip_names)))

        assert self.batch_size <= self.num_samples
    
    def get_context_window(self, clip_name, win_length, clip_length=128):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        left_context_feats = np.zeros([win_length, self.visual_feature_dim], dtype=np.float32)
        right_context_feats = np.zeros([win_length, self.visual_feature_dim], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path + clip_name)
        last_right_feat = np.load(self.sliding_clip_path + clip_name)

        for k in range(win_length):
            left_context_start = start - clip_length * (k + 1)
            left_context_end = start - clip_length * k
            right_context_start = end + clip_length * k
            right_context_end = end + clip_length * (k + 1)
            left_context_name = '{}_{}_{}.npy'.format(movie_name, left_context_start, left_context_end)
            right_context_name = '{}_{}_{}.npy'.format(movie_name, right_context_start, right_context_end)

            if os.path.exists(self.sliding_clip_path + left_context_name):
                left_context_feat = np.load(self.sliding_clip_path + left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat

            if os.path.exists(self.sliding_clip_path + right_context_name):
                right_context_feat = np.load(self.sliding_clip_path + right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat

            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return left_context_feats, right_context_feats
    
    def load_movie_byclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []

        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))

        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:        
                # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                visual_feature_path = self.sliding_clip_path + self.sliding_clip_names[k]
                feature_data = np.load(visual_feature_path)
                movie_clip_featmap.append((self.sliding_clip_names[k], feature_data))

        return movie_clip_featmap, movie_clip_sentences
    
    def load_movie_slidingclip(self, movie_name, sample_num):
        
        movie_clip_sentences = []
        movie_clip_featmap = []
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))

        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                visual_feature_path = self.sliding_clip_path + self.sliding_clip_names[k]
                left_context_feat, right_context_feat = self.get_context_window(self.sliding_clip_names[k], 1)
                feature_data = np.load(visual_feature_path)
                #comb_feat = np.hstack((left_context_feat, feature_data, right_context_feat))
                left_context_feat = np.transpose(left_context_feat, [1,0])
                right_context_feat = np.transpose(right_context_feat, [1,0])
                comb_feat = np.column_stack((left_context_feat, feature_data, right_context_feat))   
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))

        return movie_clip_featmap, movie_clip_sentences

class movieDataset(object):

    def __init__(self, movie_dir):
        
        self.sliding_clip_path = movie_dir
        self.visual_feature_dim = 1024
        self.sliding_clip_names = []
        
        movie_names_set = set()
        sliding_clips_tmp = os.listdir(movie_dir)
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[1]=="npy":
                movie_name = clip_name.split("_")[0] 
                if not movie_name in self.sliding_clip_names:
                    movie_names_set.add(movie_name)
                self.sliding_clip_names.append(clip_name.split(".")[0] + "." + clip_name.split(".")[1])

        self.movie_names = list(movie_names_set)
        print("movie number: " + str(len(self.movie_names)))
        print("sliding clips number: " + str(len(self.sliding_clip_names)))

    def load_movie_slidingclip(self, movie_name):
        
        movie_clip_featmap = []
        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:

                visual_feature_path = self.sliding_clip_path + self.sliding_clip_names[k]
                left_context_feat, right_context_feat = self.get_context_window(self.sliding_clip_names[k], 1)
                feature_data = np.load(visual_feature_path)
                left_context_feat = np.transpose(left_context_feat, [1,0])
                right_context_feat = np.transpose(right_context_feat, [1,0])
                comb_feat = np.column_stack((left_context_feat, feature_data, right_context_feat)) 
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))

        return movie_clip_featmap

    def get_context_window(self, clip_name, win_length, clip_length=128):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        left_context_feats = np.zeros([win_length, self.visual_feature_dim], dtype=np.float32)
        right_context_feats = np.zeros([win_length, self.visual_feature_dim], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path + clip_name)
        last_right_feat = np.load(self.sliding_clip_path + clip_name)

        for k in range(win_length):
            left_context_start = start - clip_length * (k + 1)
            left_context_end = start - clip_length * k
            right_context_start = end + clip_length * k
            right_context_end = end+clip_length * (k + 1)
            left_context_name = '{}_{}_{}.npy'.format(movie_name, left_context_start, left_context_end)
            right_context_name = '{}_{}_{}.npy'.format(movie_name, right_context_start, right_context_end)

            if os.path.exists(self.sliding_clip_path + left_context_name):
                left_context_feat = np.load(self.sliding_clip_path + left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat

            if os.path.exists(self.sliding_clip_path + right_context_name):
                right_context_feat = np.load(self.sliding_clip_path + right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat

            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return left_context_feats, right_context_feats
