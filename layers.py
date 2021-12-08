import tensorflow.compat.v1 as tf

def batch_normalization(x, decay=0.99, epslion=1e-3, training=True, name=None):

    _, _, _, in_channels = [i for i in x.get_shape()]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        gamma = tf.get_variable('gamma', [in_channels], initializer=tf.constant_initializer(1.0))
        shift = tf.get_variable('shift', [in_channels], initializer=tf.constant_initializer(0.0))

        moving_mean = tf.get_variable('moving_mean', [in_channels], 
                                    initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable('moving_var', [in_channels], 
                                    initializer=tf.constant_initializer(1.0), trainable=False)

        mean, var = tf.nn.moments(x, axes=[0, 1, 2])
        update_moving_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1.0 - decay))
        update_moving_var = tf.assign(moving_var, moving_var * decay + var * (1.0 - decay))
    
    if training:
        with tf.control_dependencies([update_moving_mean, update_moving_var]):      
            return tf.nn.batch_normalization(x, mean, var, shift, gamma, epslion)
    
    return tf.nn.batch_normalization(x, moving_mean, moving_var, shift, gamma, epslion)

def conv_layer(x, filter_size, ksize=3, stride=1, use_bias=True, batch_norm=False, 
                relu=True, training=True, name=None):

    _, _, _, in_channels = [i for i in x.get_shape()]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
  
        kernel = tf.get_variable('weights', [ksize, ksize, in_channels, filter_size], 
                                initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding='SAME')

        if batch_norm:
            conv = batch_normalization(conv, training=training, name='batch_norm')
        
        elif use_bias:
            biases = tf.get_variable('biases', [filter_size], 
                                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        if relu:
            conv = tf.nn.relu(conv)
        
    return conv

def convt_layer(x, filter_size, ksize=3, stride=1, use_bias=True, batch_norm=False, 
                relu=True, training=True, name=None):

    _, _, _, in_channels = [i for i in x.get_shape()]
    batch_size = tf.shape(x)[0]
    output_h = tf.shape(x)[1] * stride
    output_w = tf.shape(x)[2] * stride
    output_shape = [batch_size, output_h, output_w, filter_size]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
  
        kernel = tf.get_variable('weights', [ksize, ksize, filter_size, in_channels], 
                                initializer=tf.random_normal_initializer(stddev=tf.sqrt(2.0 / filter_size)))      
        conv = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, stride, stride, 1], padding='SAME')

        if batch_norm:
            conv = batch_normalization(conv, training=training, name='batch_norm')
        
        elif use_bias:
            biases = tf.get_variable('biases', [filter_size], 
                                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        if relu:
            conv = tf.nn.relu(conv)
        
    return conv

def fc_layer(x, filter_size, use_bias=True, relu=True, name=None):

    shape = x.get_shape()
    nodes = 1
    for i in shape[1:]:
        nodes *= i.value

    flat_x = tf.reshape(x, [-1, nodes])
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    
        fc = tf.get_variable('weights', [nodes, filter_size], 
                            initializer=tf.random_normal_initializer())

        if use_bias:
            fcb = tf.get_variable('biases', [filter_size], 
                                initializer=tf.constant_initializer(0.0))
            fc = tf.nn.bias_add(tf.matmul(flat_x, fc), fcb)

        if relu:
            fc = tf.nn.relu(fc)
    
    return fc

def res_layer(x, filter_size, use_bias=True, batch_norm=False, training=True, name=None):

    _, _, _, in_channels = [i for i in x.get_shape()]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if in_channels != filter_size:

            x = conv_layer(x, filter_size, ksize=1, use_bias=use_bias, relu=False, training=training)

        if batch_norm:
            x = tf.nn.relu(batch_normalization(x, training=training, name='batch_norm'))

        temp = conv_layer(x, filter_size, use_bias=use_bias, relu=False, training=training)
        temp = conv_layer(temp, filter_size, use_bias=use_bias, relu=False, training=training)
    
    return x + temp

def bottleneck(x, filter_size, output_size, use_bias=True, batch_norm=False, training=True, name=None):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if batch_norm:
            x = tf.nn.relu(batch_normalization(x, training=training))

        temp = conv_layer(x, filter_size, use_bias=use_bias, relu=False, training=training)
        temp = conv_layer(temp, filter_size, use_bias=use_bias, relu=False, training=training)
        temp = conv_layer(temp, output_size, ksize=1, use_bias=use_bias, relu=False, training=training)
        
    return x + temp

def dw_conv_layer(x, filter_size, ksize=3, stride=1, use_bias=True, batch_norm=False, 
                        relu=True, training=True, name=None):

    _, _, _, in_channels = [i for i in x.get_shape()]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        kernel = tf.get_variable('weights', [ksize, ksize, in_channels, filter_size], 
                                tf.random_normal_initializer(stddev=tf.sqrt(2.0 / filter_size)))      
        conv = tf.nn.depthwise_conv2d(x, kernel, [1, stride, stride, 1], padding='SAME')

        if batch_norm:
            conv = batch_normalization(conv, is_training=training, name='bn')
        
        elif use_bias:
            biases = tf.get_variable('biases', [filter_size], 
                                    tf.constant_initializer(0.0), trainable=training)
            conv = tf.nn.bias_add(conv, biases)

        if relu:
            conv = tf.nn.relu(conv)
        
    return conv

def SEBlock(x, r=16, name='SE_Block'):

    _, H, W, in_channels = [i for i in x.get_shape()]

    with tf.variable_scope(name):

        avgpool = tf.nn.avg_pool(x, ksize=[H,W], strides=1, padding='SAME', name='avg_pool')
        #maxpool = tf.nn.max_pool(x, ksize=[H,W], strides=1, padding='SAME', name='max_pool')

        fc_in = tf.reshape(avgpool, [-1, in_channels])
        fc1 = fc_layer(fc_in, in_channels / r, name='fc1')
        fc2 = fc_layer(fc1, in_channels, relu=False, name='fc2')
        fc2 = tf.math.sigmoid(fc2)

    return x * fc2

def CBAMBlock(x, r=16, name='CBAM_Block'):

    _, H, W, in_channels = [i for i in x.get_shape()]

    with tf.variable_scope(name):

        avgpool_c = tf.nn.avg_pool(x, ksize=[H,W], strides=1, padding='SAME', name='avgpool_c')
        maxpool_c = tf.nn.max_pool(x, ksize=[H,W], strides=1, padding='SAME', name='maxpool_c')

        avg_conv = conv_layer(avgpool_c, in_channels / r, use_bias=False, name='layer1')
        avg_conv = conv_layer(avg_conv, in_channels, use_bias=False, relu=False, name='layer2')
        max_conv = conv_layer(maxpool_c, in_channels / r, use_bias=False, name='layer1')
        max_conv = conv_layer(max_conv, in_channels, use_bias=False, relu=False, name='layer2')

        channel_out = tf.math.sigmoid(avg_conv + max_conv, name='channel_out')

        avgpool_s = tf.reduce_max(x, axis=3, name='spatital_avgpool')
        maxpool_s = tf.reduce_mean(x, axis=3, name='spatital_maxpool')

        spatital_conv = conv_layer(tf.concat(avgpool_s, maxpool_s), 1, relu=False)

        spatital_out = tf.math.sigmoid(spatital_conv, name='spatital_out')

    return x + (x * channel_out * spatital_out)

def SABlock(queries, keys, values, d_k, d_v, d_model, head, rata=.1, 
            mask=None, weights=None, training=True, name='Self_Attention'):

    N, nq = queries.shape[:2]
    nk = keys.shape[1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        q = fc_layer(queries, head * d_k, relu=False)  
        k = fc_layer(keys, head * d_k, relu=False)  
        v = fc_layer(values, head * d_v, relu=False)  

        q = tf.reshape(q, [N, nq, head, d_k], perm=[0, 2, 1, 3]) # (N, h, nq, d_k)
        k = tf.reshape(q, [N, nk, head, d_k], perm=[0, 2, 3, 1]) # (N, h, d_k, nk)
        v = tf.reshape(q, [N, nk, head, d_v], perm=[0, 2, 1, 3]) # (N, h, nk, d_v)

        att = tf.matmul(q, k) / tf.sqrt(d_k)  # (N, h, nq, nk)
        if weights is not None:
            att = att * weights
        if mask is not None:
            att = att.masked_fill(mask, -tf.inf)

        att = tf.nn.softmax(att)
        att = tf.nn.dropout(att, rata, training=training)

        out = tf.matmul(att, v)
        out = fc_layer(out, d_model)  # (b_s, nq, d_model)

        return out