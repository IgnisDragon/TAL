import os, math
import numpy as np
import tensorflow as tf
from data import dataset

_IMAGE_SIZE = 224
_NUM_CLASSES = 600
_FRAMES = 16

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
    'custom': 'models/model.ckpt',
}

def Unit3D(inputs, 
           out_channels, 
           kernel_size=[1, 1, 1], 
           stride=[1, 1, 1], 
           activation='relu',
           padding='SAME', 
           use_bias=False, 
           use_bn=True,
           training=True,
           name=None):
           
    with tf.variable_scope(name):

        conv3d = tf.layers.Conv3D(out_channels, 
                                  kernel_size=kernel_size, 
                                  strides=stride, 
                                  padding='SAME',
                                  use_bias=use_bias)(inputs)

        if use_bn:
            conv3d = tf.layers.batch_normalization(conv3d, training=training)

        if activation == 'relu':
            conv3d = tf.nn.relu(conv3d)

    return conv3d

def Mixed(x, out_channels, training=True, name=None):

    with tf.variable_scope(name):

        with tf.variable_scope('Branch_0'):
            branch_0 = Unit3D(x, out_channels[0], training=training, name='Conv3d_0a_1x1')

        with tf.variable_scope('Branch_1'):
            branch_1 = Unit3D(x, out_channels[1], training=training, name='Conv3d_0a_1x1')
            branch_1 = Unit3D(branch_1, out_channels[2], kernel_size=[3, 3, 3], training=training, name='Conv3d_0b_3x3')

        with tf.variable_scope('Branch_2'):
            branch_2 = Unit3D(x, out_channels[3], training=training, name='Conv3d_0a_1x1')
            branch_2 = Unit3D(branch_2, out_channels[4], kernel_size=[3, 3, 3], training=training, name='Conv3d_0b_3x3')

        with tf.variable_scope('Branch_3'):
            branch_3 = tf.nn.max_pool3d(x, ksize=[3, 3, 3], strides=[1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
            branch_3 = Unit3D(branch_3, out_channels[5], training=training, name='Conv3d_0b_1x1')
        
    return tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

class net:

    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits',
                 ckpt_dir=None, save_dir=None):
        
        #self.sess = tf.Session()
        self.ckpt_dir = ckpt_dir
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze
        self.final_endpoint = final_endpoint
        
        self.VALID_ENDPOINTS = (
            'Conv3d_1a_7x7',
            'MaxPool3d_2a_3x3',
            'Conv3d_2b_1x1',
            'Conv3d_2c_3x3',
            'MaxPool3d_3a_3x3',
            'Mixed_3b',
            'Mixed_3c',
            'MaxPool3d_4a_3x3',
            'Mixed_4b',
            'Mixed_4c',
            'Mixed_4d',
            'Mixed_4e',
            'Mixed_4f',
            'MaxPool3d_5a_2x2',
            'Mixed_5b',
            'Mixed_5c',
            'Logits',
            'Predictions',
        )
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
    
    def i3d(self, x, rate, training=True, name='i3d'):
        # rate: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.
        end_points = {}
        with tf.variable_scope(name):

            conv3d = Unit3D(x, 64, kernel_size=[7, 7, 7], stride=[2, 2, 2], training=training, name='Conv3d_1a_7x7')
            end_points['Conv3d_1a_7x7'] = conv3d
            if self.final_endpoint == 'Conv3d_1a_7x7': return conv3d, end_points

            conv3d = tf.nn.max_pool3d(conv3d, ksize=[1, 3, 3], strides=[1, 2, 2], padding='SAME', name='MaxPool3d_2a_3x3')
            end_points['MaxPool3d_2a_3x3'] = conv3d
            if self.final_endpoint == 'MaxPool3d_2a_3x3': return conv3d, end_points

            conv3d = Unit3D(conv3d, 64, kernel_size=[1, 1, 1], training=training, name='Conv3d_2b_1x1')
            end_points['Conv3d_2b_1x1'] = conv3d
            if self.final_endpoint == 'Conv3d_2b_1x1': return conv3d, end_points

            conv3d = Unit3D(conv3d, 192, kernel_size=[3, 3, 3], training=training, name='Conv3d_2c_3x3')
            end_points['Conv3d_2c_3x3'] = conv3d
            if self.final_endpoint == 'Conv3d_2c_3x3': return conv3d, end_points

            conv3d = tf.nn.max_pool3d(conv3d, ksize=[1, 3, 3], strides=[1, 2, 2], padding='SAME', name='MaxPool3d_3a_3x3')
            end_points['MaxPool3d_3a_3x3'] = conv3d
            if self.final_endpoint == 'MaxPool3d_3a_3x3': return conv3d, end_points

            mixed = Mixed(conv3d, [64, 96, 128, 16, 32, 32], training=training, name='Mixed_3b')
            end_points['Mixed_3b'] = mixed
            if self.final_endpoint == 'Mixed_3b': return mixed, end_points

            mixed = Mixed(mixed, [128, 128, 192, 32, 96, 64], training=training, name='Mixed_3c')
            end_points['Mixed_3c'] = mixed
            if self.final_endpoint == 'Mixed_3c': return mixed, end_points

            mixed = tf.nn.max_pool3d(mixed, ksize=[3, 3, 3], strides=[2, 2, 2], padding='SAME', name='MaxPool3d_4a_3x3')
            end_points['MaxPool3d_4a_3x3'] = mixed
            if self.final_endpoint == 'MaxPool3d_4a_3x3': return mixed, end_points

            mixed = Mixed(mixed, [192, 96, 208, 16, 48, 64], training=training, name='Mixed_4b')
            end_points['Mixed_4b'] = mixed
            if self.final_endpoint == 'Mixed_4b': return mixed, end_points

            mixed = Mixed(mixed, [160, 112, 224, 24, 64, 64], training=training, name='Mixed_4c')
            end_points['Mixed_4c'] = mixed
            if self.final_endpoint == 'Mixed_4c': return mixed, end_points

            mixed = Mixed(mixed, [128, 128, 256, 24, 64, 64], training=training, name='Mixed_4d')
            end_points['Mixed_4d'] = mixed
            if self.final_endpoint == 'Mixed_4d': return mixed, end_points

            mixed = Mixed(mixed, [112, 144, 288, 32, 64, 64], training=training, name='Mixed_4e')
            end_points['Mixed_4e'] = mixed
            if self.final_endpoint == 'Mixed_4e': return mixed, end_points

            mixed = Mixed(mixed, [256, 160, 320, 32, 128, 128], training=training, name='Mixed_4f')
            end_points['Mixed_4f'] = mixed
            if self.final_endpoint == 'Mixed_4f': return mixed, end_points

            mixed = tf.nn.max_pool3d(mixed, ksize=[2, 2, 2], strides=[2, 2, 2], padding='SAME', name='MaxPool3d_5a_2x2')
            end_points['MaxPool3d_5a_2x2'] = mixed
            if self.final_endpoint == 'MaxPool3d_5a_2x2': return mixed, end_points

            mixed = Mixed(mixed, [256, 160, 320, 32, 128, 128], training=training, name='Mixed_5b')
            end_points['Mixed_5b'] = mixed
            if self.final_endpoint == 'Mixed_5b': return mixed, end_points

            mixed = Mixed(mixed, [384, 192, 384, 48, 128, 128], training=training, name='Mixed_5c')
            end_points['Mixed_5c'] = mixed
            if self.final_endpoint == 'Mixed_5c': return mixed, end_points

            with tf.variable_scope('Logits'):

                net = tf.nn.avg_pool3d(mixed, ksize=[2, 7, 7], strides=[1, 1, 1], padding='VALID')
                net = tf.nn.dropout(net, rate=rate)
                logits = Unit3D(net, self.num_classes, kernel_size=[1, 1, 1], use_bias=True, use_bn=False, 
                                training=training, name='Conv3d_0c_1x1')
                
                if self.spatial_squeeze:
                    logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')

            averaged_logits = tf.reduce_mean(logits, axis=1)
            end_points['Logits'] = averaged_logits
            if self.final_endpoint == 'Logits': return averaged_logits, end_points

            predictions = tf.nn.softmax(averaged_logits)
            end_points['predictions'] = predictions

        return predictions, end_points

def extract_feature(batch_size=1, sample_size=64, overlap=0.8):

    save_dir = 'E:/File/VS Code/DataSet/TACoS/Interval64_128_i3d_mixed5/'
    movie_dir = 'E:/File/VS Code/DataSet/TACoS/MPII-Cooking-2-videos'
    movie_file = '../dataset/convert_list.txt'

    movie_data = dataset(movie_dir, movie_file, batch_size)

    with tf.Session() as sess:
    
        inputs = tf.placeholder(tf.float32, shape=(batch_size, _FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

        model = net(_NUM_CLASSES, final_endpoint='Mixed_5c')
        logits , _ = model.i3d(inputs, 0.0, False)
        logits = tf.nn.avg_pool3d(logits, ksize=[2, 7, 7], strides=[1, 1, 1], padding='VALID')
        logits = tf.squeeze(logits, [1, 2, 3])

        saver = tf.train.Saver()
        saver.restore(sess, _CHECKPOINT_PATHS['custom'])

        next_start = 0
        next = 0
        flag = -1
        while next_start < len(movie_data):
            
            data, name, next_start, next = movie_data.next_batch(next_start, next, sample_size, overlap)

            if next_start >= len(movie_data): break
            
            feature = sess.run(logits, feed_dict={inputs:data})
            
            for i in range(next - len(feature), next):
                num = int(i * sample_size * (1 - overlap)) + 1
                path = save_dir + name + '_' + str(num) + '_' + str(num + sample_size)
                np.save(path, feature[i - next])
            
            if flag != next_start:
                flag = next_start
                if next_start != len(movie_data): print(name)
    
    print('complete')
        
if __name__ == '__main__':
    extract_feature(sample_size=64)
            