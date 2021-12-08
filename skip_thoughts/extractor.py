import os
import numpy as np
import skipthoughts as skipthoughts
from tqdm import tqdm
import pickle
import time

sent_path = './dataset/corpus/annosDetailed-processed.csv'
train_file_path = './dataset/TACoS_test.txt'
encoder = skipthoughts.Encoder(skipthoughts.load_model())

sent_embed = {}
data = np.genfromtxt(sent_path, delimiter='\t', usecols=(0,6,7,8), 
                    dtype=None, encoding='utf-8')
anno_size = len(data)

movie_name = set()
with open(train_file_path) as file:
    for line in file:  
        movie_name.add(line.split('-')[0] + '-' + line.split('-')[1])

progress = tqdm(total=anno_size, initial=0)
start = time.time()
""""
for name, sent, start_time, end_time in data:
    sent_name = '{}_{}_{}'.format(name, start_time, end_time)
    #progress.set_description(sent_name)
    if name in movie_name:
        if sent_name not in sent_embed:
            sent_embed[sent_name] = []
        sent_vec = encoder.encode([sent])
        sent_embed[sent_name].append(sent_vec)
    #progress.update(1)
"""
tmp_name = ''
tmp_vec = []
for name, sent, start_time, end_time in data:
    sent_name = '{}_{}_{}'.format(name, start_time, end_time)
    progress.set_description(sent_name)
    if name in movie_name:
        if tmp_name != sent_name:
            if len(tmp_vec) != 0:
                sent_vec = encoder.encode(tmp_vec)
                sent_embed[sent_name] = sent_vec
            tmp_name = sent_name
            tmp_vec = []
        tmp_vec.append(sent)
    progress.update(1)

print('-time: ' + str(time.time() - start))
with open('./exp_data/test_sentence.pkl', 'wb') as handle:
    pickle.dump(sent_embed, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("query number: " + str(len(sent_embed)))