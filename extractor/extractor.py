import numpy as np
import time
from tqdm import tqdm
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from input_data import *
import model_c3d
import model_i3d

def extractor_c3d(batch_size, points='fc1'):

    inputs = tf.placeholder(tf.float32, shape=(batch_size, model_c3d.FRAMES_SIZE, 
                                                model_c3d.CROP_SIZE, model_c3d.CROP_SIZE, 3))

    logits, _ = model_c3d.c3d(inputs, batch_size, final_endpoint=points)

    return logits, inputs

def extractor_i3d(batch_size, points='Mixed_5c'):

    inputs = tf.placeholder(tf.float32, shape=(batch_size, model_i3d.FRAMES_SIZE, 
                                                model_i3d.CROP_SIZE, model_i3d.CROP_SIZE, 3))

    logits, _ = model_i3d.i3d(inputs, batch_size, final_endpoint=points)
    logits = tf.nn.avg_pool3d(logits, ksize=[2, 7, 7], strides=[1, 1, 1], padding='VALID')
    logits = tf.squeeze(logits, [1, 2, 3])

    return logits, inputs

def feature_extract(model_name, batch_size=1, sample_size=64, overlap=0.8):

    save_dir = './extractor/feature/Interval' + str(sample_size) + '_' + model_name + '/'
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    c3d_ckpt = "./extractor/models/conv3d_deepnetA_sport1m_iter_1900000_TF.model"
    i3d_ckpt = "./extractor/models/i3d_kinetics600.ckpt"

    movie_dir = 'E:/File/VS Code/DataSet/TACoS/videos'
    movie_file = './dataset/TACoS_train.txt'
    
    model = None
    movie_data = None
    if model_name=='c3d': 
        logits, inputs = extractor_c3d(batch_size)
        model = c3d_ckpt
        movie_data = dataset(movie_dir, movie_file, batch_size, 
                            frames_size=model_c3d.FRAMES_SIZE,
                            crop_size=model_c3d.CROP_SIZE, 
                            crop_mean=True)

    if model_name=='i3d': 
        logits, inputs = extractor_i3d(batch_size)
        model = i3d_ckpt
        movie_data = dataset(movie_dir, movie_file, batch_size, 
                            frames_size=model_i3d.FRAMES_SIZE,
                            crop_size=model_i3d.CROP_SIZE, 
                            rescale=True)

    for var in tf.trainable_variables(): print(var.name)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, model)

        next_start = 0
        next = 0
        flag = -1
        count = 0
        num = 1
        progress = tqdm(total=len(movie_data))

        while True:

            data, name, next_start, next = movie_data.next_batch(next_start, next, sample_size, overlap)

            if next_start >= len(movie_data): break

            if flag != next_start:
                flag = next_start
                count = 0
                num = 1
                progress.update(1)

            feature = sess.run(logits, feed_dict={inputs:data})

            for i in range(count, count + batch_size):
                path = save_dir + name + '_' + str(num) + '_' + str(num + sample_size)
                progress.set_description(name + '_' + str(num) + '_' + str(num + sample_size))
                num += int(sample_size * (1 - overlap))
                np.save(path, feature[i - next])
                
            count += batch_size

if __name__ == '__main__':
    feature_extract('i3d', sample_size=64)