import tensorflow.compat.v1 as tf
import input_data
import c3d_model
import numpy as np
import time

tf.disable_v2_behavior()

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           c3d_model.NUM_FRAMES_PER_CLIP,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))

    return images_placeholder, labels_placeholder

def variable_with_weight_decay(name, shape, stddev, wd, initializer=None):
    
    if initializer == None: 
        initializer = tf.truncated_normal_initializer(stddev=stddev)

    var = tf.get_variable(name, shape, initializer)

    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)

    return var

def variable_weight(name, shape, stddev=0.04, initializer=None):

    if initializer == None: 
        initializer = tf.truncated_normal_initializer(stddev=stddev)

    var = tf.get_variable(name, shape, initializer=initializer)

    return var

def feature_extract(batch_size=1, extract=None):

    model_name = "./models/conv3d_deepnetA_sport1m_iter_1900000_TF.model"
    test_list_file = './dataset/test.txt'

    num_test_videos = len(list(open(test_list_file,'r')))
    print("Number of test videos={}".format(num_test_videos))

    # Get the sets of images and labels for training, validation, and
    images_placeholder, _ = placeholder_inputs(batch_size)

    with tf.variable_scope('var_name') as var_scope:
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
            'out': variable_weight('wout', [4096, c3d_model.NUM_CLASSES])
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
            'out': variable_weight('bout', [c3d_model.NUM_CLASSES]),
        }

    logits = c3d_model.inference_c3d(images_placeholder, batch_size, weights, biases)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    gpu_options.allow_soft_placement=True
    sess = tf.Session(config=gpu_options)

    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name)

    #vars = tf.trainable_variables()
    #print(vars) #some infos about variables...
    #vars_vals = sess.run(vars)
    #for var, val in zip(vars, vars_vals):
        #print(var.name)
        #print("var: {}, value: {}".format(var.name, val)) #...or sort it in a list....

    # And then after everything is built, start the training loop.
    next_start_pos = 0
    all_steps = int((num_test_videos - 1) / batch_size) + 1

    for step in range(all_steps):

        start_time = time.time()
        test_images, next_start_pos, _, valid_len = \
            input_data.read_clip(
                test_list_file,
                batch_size,
                start_pos=next_start_pos,
                num_frames_per_clip=128
            )
        _, variables = sess.run(logits, feed_dict={images_placeholder:test_images})
        
        if extract != None:
            feature = variables.get(extract)
            np.save(extract + '_' + str(step), feature)

    print("done")

if __name__ == '__main__':

    feature_extract(extract='fc1')