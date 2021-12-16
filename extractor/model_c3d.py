import tensorflow.compat.v1 as tf

# UCF-101 dataset
#NUM_CLASSES = 101
# sports-1M
NUM_CLASSES = 487
CROP_SIZE = 112
FRAMES_SIZE = 16

def variable_weight(name, shape, stddev=0.04, initializer=None):

    if initializer == None: 
        initializer = tf.truncated_normal_initializer(stddev=stddev)

    var = tf.get_variable(name, shape, initializer=initializer)

    return var
	
def conv3d(inputs, w, b, name=None):

	with tf.variable_scope(name):

		conv3d = tf.nn.conv3d(inputs, w, strides=[1, 1, 1, 1, 1], padding='SAME')
		conv3d = tf.nn.bias_add(conv3d, b)
	
	return conv3d

def max_pool(inputs, ksize, name=None):
	return tf.nn.max_pool3d(inputs, ksize=[1, ksize, 2, 2, 1], strides=[1, ksize, 2, 2, 1], padding='SAME', name=name)

def c3d(X, batch_size, final_endpoint='fc1'):

	with tf.variable_scope('var_name'):
		weights = {
			'wc1': variable_weight('wc1', [3, 3, 3, 3, 64]),
			'wc2': variable_weight('wc2', [3, 3, 3, 64, 128]),
			'wc3a': variable_weight('wc3a', [3, 3, 3, 128, 256]),
			'wc3b': variable_weight('wc3b', [3, 3, 3, 256, 256]),
			'wc4a': variable_weight('wc4a', [3, 3, 3, 256, 512]),
			'wc4b': variable_weight('wc4b', [3, 3, 3, 512, 512]),
			'wc5a': variable_weight('wc5a', [3, 3, 3, 512, 512]),
			'wc5b': variable_weight('wc5b', [3, 3, 3, 512, 512]),
			'wd1': variable_weight('wd1', [8192, 4096]),
			'wd2': variable_weight('wd2', [4096, 4096]),
			'out': variable_weight('wout', [4096, NUM_CLASSES])
		}
		biases = {
			'bc1': variable_weight('bc1', [64]),
			'bc2': variable_weight('bc2', [128]),
			'bc3a': variable_weight('bc3a', [256]),
			'bc3b': variable_weight('bc3b', [256]),
			'bc4a': variable_weight('bc4a', [512]),
			'bc4b': variable_weight('bc4b', [512]),
			'bc5a': variable_weight('bc5a', [512]),
			'bc5b': variable_weight('bc5b', [512]),
			'bd1': variable_weight('bd1', [4096]),
			'bd2': variable_weight('bd2', [4096]),
			'out': variable_weight('bout', [NUM_CLASSES]),
		}

	variables = {}
	# Convolution Layer
	conv1 = conv3d(X, weights['wc1'], biases['bc1'], name='conv1')
	conv1 = tf.nn.relu(conv1, name='relu1')
	variables['conv1'] = conv1
	if final_endpoint == 'conv1': return conv1, variables

	pool1 = max_pool(conv1, ksize=1, name='pool1')
	variables['pool1'] = pool1
	if final_endpoint == 'pool1': return pool1, variables

	# Convolution Layer
	conv2 = conv3d(pool1, weights['wc2'], biases['bc2'], name='conv2')
	conv2 = tf.nn.relu(conv2, name='relu2')
	variables['conv2'] = conv2
	if final_endpoint == 'conv2': return conv2, variables

	pool2 = max_pool(conv2, ksize=2, name='pool2')
	variables['pool2'] = pool2
	if final_endpoint == 'pool2': return pool2, variables

	# Convolution Layer
	conv3 = conv3d(pool2, weights['wc3a'], biases['bc3a'], name='conv3a')
	conv3 = tf.nn.relu(conv3, name='relu3a')
	conv3 = conv3d(conv3, weights['wc3b'], biases['bc3b'], name='conv3b')
	conv3 = tf.nn.relu(conv3, name='relu3b')
	variables['conv3'] = conv3
	if final_endpoint == 'conv3': return conv3, variables

	pool3 = max_pool(conv3, ksize=2, name='pool3')
	variables['pool3'] = pool3
	if final_endpoint == 'pool3': return pool3, variables

	# Convolution Layer
	conv4 = conv3d(pool3, weights['wc4a'], biases['bc4a'], name='conv4a')
	conv4 = tf.nn.relu(conv4, name='relu4a')
	conv4 = conv3d(conv4, weights['wc4b'], biases['bc4b'], name='conv4b')
	conv4 = tf.nn.relu(conv4, 'relu4b')
	variables['conv4'] = conv4
	if final_endpoint == 'conv4': return conv4, variables

	pool4 = max_pool(conv4, ksize=2, name='pool4')	
	variables['pool4'] = pool4
	if final_endpoint == 'pool4': return pool4, variables

	# Convolution Layer
	conv5 = conv3d(pool4, weights['wc5a'], biases['bc5a'], name='conv5a')
	conv5 = tf.nn.relu(conv5, name='relu5a')
	conv5 = conv3d(conv5, weights['wc5b'], biases['bc5b'], name='conv5b')
	conv5 = tf.nn.relu(conv5, name='relu5b')
	variables['conv5'] = conv5
	if final_endpoint == 'conv5': return conv5, variables

	pool5 = max_pool(conv5, ksize=2, name='pool5')
	variables['pool5'] = pool5
	if final_endpoint == 'pool5': return pool5, variables
	# Fully connected layer
	# if load conv3d_deepnetA_sport1m_iter_1900000_TF.model or c3d_ucf101_finetune_whole_iter_20000_TF.model,
	# you don't need tranpose operation,just comment that line code.
	#pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
	dense1 = tf.reshape(pool5, [batch_size, weights['wd1'].get_shape().as_list()[0]]) 
	dense1 = tf.matmul(dense1, weights['wd1']) + biases['bd1']

	dense1 = tf.nn.relu(dense1, name='fc1')
	#dense1 = tf.nn.dropout(dense1, _dropout)
	variables['fc1'] = dense1
	if final_endpoint == 'fc1': return dense1, variables

	dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2')
	#dense2 = tf.nn.dropout(dense2, _dropout)
	variables['fc2'] = dense2
	if final_endpoint == 'fc2': return dense2, variables

	# Output] =  class prediction
	out = tf.matmul(dense2, weights['out']) + biases['out']
	variables['out'] = out

	return out, variables
