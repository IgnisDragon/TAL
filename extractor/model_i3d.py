import tensorflow.compat.v1 as tf

NUM_CLASSES = 400
CROP_SIZE = 224
FRAMES_SIZE = 16

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
                                  padding=padding,
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

VALID_ENDPOINTS = (
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
    
def i3d(x, training=False, spatial_squeeze=True, final_endpoint='Logits'):

    if final_endpoint not in VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

    end_points = {}
    with tf.variable_scope('i3d'):
        conv3d = Unit3D(x, 64, kernel_size=[7, 7, 7], stride=[2, 2, 2], training=training, name='Conv3d_1a_7x7')
        end_points['Conv3d_1a_7x7'] = conv3d
        if final_endpoint == 'Conv3d_1a_7x7': return conv3d, end_points

        conv3d = tf.nn.max_pool3d(conv3d, ksize=[1, 3, 3], strides=[1, 2, 2], padding='SAME', name='MaxPool3d_2a_3x3')
        end_points['MaxPool3d_2a_3x3'] = conv3d
        if final_endpoint == 'MaxPool3d_2a_3x3': return conv3d, end_points

        conv3d = Unit3D(conv3d, 64, kernel_size=[1, 1, 1], training=training, name='Conv3d_2b_1x1')
        end_points['Conv3d_2b_1x1'] = conv3d
        if final_endpoint == 'Conv3d_2b_1x1': return conv3d, end_points
        
        conv3d = Unit3D(conv3d, 192, kernel_size=[3, 3, 3], training=training, name='Conv3d_2c_3x3')
        end_points['Conv3d_2c_3x3'] = conv3d
        if final_endpoint == 'Conv3d_2c_3x3': return conv3d, end_points

        conv3d = tf.nn.max_pool3d(conv3d, ksize=[1, 3, 3], strides=[1, 2, 2], padding='SAME', name='MaxPool3d_3a_3x3')
        end_points['MaxPool3d_3a_3x3'] = conv3d
        if final_endpoint == 'MaxPool3d_3a_3x3': return conv3d, end_points

        mixed = Mixed(conv3d, [64, 96, 128, 16, 32, 32], training=training, name='Mixed_3b')
        end_points['Mixed_3b'] = mixed
        if final_endpoint == 'Mixed_3b': return mixed, end_points

        mixed = Mixed(mixed, [128, 128, 192, 32, 96, 64], training=training, name='Mixed_3c')
        end_points['Mixed_3c'] = mixed
        if final_endpoint == 'Mixed_3c': return mixed, end_points

        mixed = tf.nn.max_pool3d(mixed, ksize=[3, 3, 3], strides=[2, 2, 2], padding='SAME', name='MaxPool3d_4a_3x3')
        end_points['MaxPool3d_4a_3x3'] = mixed
        if final_endpoint == 'MaxPool3d_4a_3x3': return mixed, end_points

        mixed = Mixed(mixed, [192, 96, 208, 16, 48, 64], training=training, name='Mixed_4b')
        end_points['Mixed_4b'] = mixed
        if final_endpoint == 'Mixed_4b': return mixed, end_points

        mixed = Mixed(mixed, [160, 112, 224, 24, 64, 64], training=training, name='Mixed_4c')
        end_points['Mixed_4c'] = mixed
        if final_endpoint == 'Mixed_4c': return mixed, end_points

        mixed = Mixed(mixed, [128, 128, 256, 24, 64, 64], training=training, name='Mixed_4d')
        end_points['Mixed_4d'] = mixed
        if final_endpoint == 'Mixed_4d': return mixed, end_points

        mixed = Mixed(mixed, [112, 144, 288, 32, 64, 64], training=training, name='Mixed_4e')
        end_points['Mixed_4e'] = mixed
        if final_endpoint == 'Mixed_4e': return mixed, end_points

        mixed = Mixed(mixed, [256, 160, 320, 32, 128, 128], training=training, name='Mixed_4f')
        end_points['Mixed_4f'] = mixed
        if final_endpoint == 'Mixed_4f': return mixed, end_points

        mixed = tf.nn.max_pool3d(mixed, ksize=[2, 2, 2], strides=[2, 2, 2], padding='SAME', name='MaxPool3d_5a_2x2')
        end_points['MaxPool3d_5a_2x2'] = mixed
        if final_endpoint == 'MaxPool3d_5a_2x2': return mixed, end_points
        
        mixed = Mixed(mixed, [256, 160, 320, 32, 128, 128], training=training, name='Mixed_5b')
        end_points['Mixed_5b'] = mixed
        if final_endpoint == 'Mixed_5b': return mixed, end_points

        mixed = Mixed(mixed, [384, 192, 384, 48, 128, 128], training=training, name='Mixed_5c')
        end_points['Mixed_5c'] = mixed
        if final_endpoint == 'Mixed_5c': return mixed, end_points

        with tf.variable_scope('Logits'):

            net = tf.nn.avg_pool3d(mixed, ksize=[2, 7, 7], strides=[1, 1, 1], padding='VALID')
            #net = tf.nn.dropout(net, rate=rate)
            logits = Unit3D(net, NUM_CLASSES, kernel_size=[1, 1, 1], use_bias=True, use_bn=False, 
                            training=training, name='Conv3d_0c_1x1')
            
            if spatial_squeeze:
                logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')

        averaged_logits = tf.reduce_mean(logits, axis=1)
        end_points['Logits'] = averaged_logits
        if final_endpoint == 'Logits': return averaged_logits, end_points

        predictions = tf.nn.softmax(averaged_logits)
        end_points['predictions'] = predictions

    return predictions, end_points