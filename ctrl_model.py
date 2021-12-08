import tensorflow.compat.v1 as tf

#from dataset import TestingDataSet
#from dataset import TrainingDataSet
#from dataset import movieDataset
from layers import *

class CTRL_Model(object):
    def __init__(self, batch_size=56, context_num=1, learning_rate=1e-3):
        
        self.batch_size = batch_size
        self.test_batch_size = 1
        self.learning_rate = learning_rate
        self.lambda_regression = 0.01
        self.alpha = 1.0 / batch_size
        self.context_num = context_num
        self.sentence_embedding_size = 4800 # emmbedding size
        self.visual_feature_dim = 1024 # visual feature size
        self.semantic_size = 1024 # the size of visual and semantic comparison size
        #self.train_set=TrainingDataSet(train_visual_feature_dir, train_csv_path, self.batch_size)
        #self.test_set=TestingDataSet(test_visual_feature_dir, test_csv_path, self.test_batch_size)
   
    '''
    used in training alignment model, CTRL(aln)
    '''	
    def fill_feed_dict_train(self, image_batch, sentence_batch, offset_batch):

        #image_batch, sentence_batch, offset_batch = self.train_set.next_batch()
        input_feed = {
                self.visual_feat_train: image_batch, # batch, visual, context
                self.sent_train: sentence_batch, # batch, sentence
                self.offset: offset_batch # batch, 2
        }
        return input_feed
    
    '''
    used in training alignment+regression model, CTRL(reg)
    '''
    def fill_feed_dict_train_reg(self, image_batch, sentence_batch, offset_batch):

        #image_batch, sentence_batch, offset_batch = self.train_set.next_batch_iou()
        input_feed = {
                self.visual_feat_train: image_batch, # batch, visual, context
                self.sent_train: sentence_batch, # batch, sentence
                self.offset: offset_batch # batch, 2
        }
        return input_feed

    '''
    cross modal processing module
    '''
    def cross_modal_comb(self, visual_feat, sentence_embed, batch_size):
        # 這是完成特徵交叉的模塊，會分別做加法、乘法和拼接
        # 因為影片會有多個，而句子只有一個，所以要做一下維度變化
        # [batch_size * batch_size, self.semantic_size] 
        vv_feature = tf.reshape(tf.tile(visual_feat, [batch_size, 1]),
                                [batch_size, batch_size, self.semantic_size]) 
        # [batch_size, batch_size * self.semantic_size]  
        ss_feature = tf.reshape(tf.tile(sentence_embed, [1, batch_size]), 
                                [batch_size, batch_size, self.semantic_size])
        concat_feature = tf.reshape(tf.concat([vv_feature, ss_feature], 2),
                                    [batch_size, batch_size, self.semantic_size * 2])
        #print(concat_feature.get_shape().as_list())

        mul_feature = tf.multiply(vv_feature, ss_feature) # 56,56,1024，乘法
        add_feature = tf.add(vv_feature, ss_feature) # 56,56,1024，加法
        # 將各個特徵一起合併起來得到組合特徵
        comb_feature = tf.reshape(tf.concat([mul_feature, add_feature, concat_feature], 2),
                                            [1, batch_size, batch_size, self.semantic_size * 4])
        
        return comb_feature
    
    '''
    visual semantic inference, including visual semantic alignment and clip location regression
    '''
    def visual_semantic_train(self, visual_feat, sentence_embed):

        with tf.variable_scope("CTRL_Model", reuse=tf.AUTO_REUSE):

            print("Building train network...............................")
            # 對視覺特徵投影到語義空間並norm
            transformed_clip = fc_layer(visual_feat, self.semantic_size, relu=False, name='v2s_lt') 
            transformed_clip = tf.nn.l2_normalize(transformed_clip, dim=1)
            # 對本文特徵投影到語義空間並norm
            transformed_sentence = fc_layer(sentence_embed, self.semantic_size, relu=False, name='s2s_lt')
            transformed_sentence = tf.nn.l2_normalize(transformed_sentence, dim=1)
            # 特徵交叉
            cross_modal_vec = self.cross_modal_comb(transformed_clip, transformed_sentence, self.batch_size)
            # 2層FC得到預測結果
            layer1 = conv_layer(cross_modal_vec, 1000, ksize=1, name='layer1_lt')
            layer2 = conv_layer(layer1, 3, ksize=1, relu=False, name='layer2_lt')

            sim_score_mat = tf.reshape(layer2, [self.batch_size, self.batch_size, 3])

            return sim_score_mat

    def visual_semantic_test(self, visual_feat, sentence_embed):
        
        with tf.variable_scope("CTRL_Model", reuse=True):

            print("Building test network...............................")
            transformed_clip = fc_layer(visual_feat, self.semantic_size, relu=False, name='v2s_lt') 
            transformed_clip = tf.nn.l2_normalize(transformed_clip, dim=1)
            transformed_sentence = fc_layer(sentence_embed, self.semantic_size, relu=False, name='s2s_lt')
            transformed_sentence = tf.nn.l2_normalize(transformed_sentence, dim=1)

            cross_modal_vec = self.cross_modal_comb(transformed_clip, transformed_sentence, self.test_batch_size)
            layer1 = conv_layer(cross_modal_vec, 1000, ksize=1, name='layer1_lt')
            layer2 = conv_layer(layer1, 3, ksize=1, relu=False, name='layer2_lt')
            
            sim_score_mat = tf.reshape(layer2, [3])

            return sim_score_mat

    '''
    compute alignment and regression loss
    '''
    def compute_loss_reg(self, sim_reg_mat, offset_label):
        # 3 * [batch_size, batch_size, 1]
        sim_score_mat, p_reg_mat, l_reg_mat = tf.split(sim_reg_mat, 3, 2) 
        sim_score_mat = tf.reshape(sim_score_mat, [self.batch_size, self.batch_size])   # alignment score
        l_reg_mat = tf.reshape(l_reg_mat, [self.batch_size, self.batch_size])   # left regression
        p_reg_mat = tf.reshape(p_reg_mat, [self.batch_size, self.batch_size])   # right regression
        
        I_2 = tf.diag(tf.constant(-2.0, shape=[self.batch_size])) # unit matrix with -2
        all1 = tf.constant(1.0, shape=[self.batch_size, self.batch_size])
        #               | -1  1   1...   |

        #   mask_mat =  | 1  -1  -1...   |

        #               | 1   1  -1 ...  |
        mask_mat = tf.add(I_2, all1)

        # loss cls, not considering iou
        I = tf.diag(tf.constant(1.0, shape=[self.batch_size]))
        alpha_mat = tf.constant(self.alpha, shape=[self.batch_size, self.batch_size])
        para_mat = tf.add(I, alpha_mat)
        # alignment loss
        loss_mat = tf.log(tf.add(all1, tf.exp(tf.multiply(mask_mat, sim_score_mat))))
        loss_mat = tf.multiply(loss_mat, para_mat)
        loss_align = tf.reduce_mean(loss_mat)
        # regression loss
        l_reg_diag = tf.matmul(tf.multiply(l_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        p_reg_diag = tf.matmul(tf.multiply(p_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        offset_pred = tf.concat((p_reg_diag, l_reg_diag), 1)
        loss_reg = tf.reduce_mean(tf.abs(tf.subtract(offset_pred, offset_label)))

        loss = tf.add(tf.multiply(self.lambda_regression, loss_reg), loss_align) # 0.01

        return loss, offset_pred, loss_reg

    def init_placeholder(self):

        visual_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim, 2 * self.context_num + 1))
        sentence_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.sentence_embedding_size))
        offset = tf.placeholder(tf.float32, shape=(self.batch_size,2))
        visual_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim, 2 * self.context_num + 1))
        sentence_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.sentence_embedding_size))

        return visual_train, sentence_train, offset, visual_test, sentence_test
    
    def get_variables_by_name(self, name_list):

        v_list = tf.trainable_variables()
        v_dict = {}
        for name in name_list:
            v_dict[name] = []
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        for name in name_list:
            print("Variables of <{}>".format(name))
            for v in v_dict[name]:
                print(v)

        return v_dict

    def training(self, loss):
        
        v_dict = self.get_variables_by_name(["lt"])
        vs_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        vs_train_op = vs_optimizer.minimize(loss, var_list=v_dict["lt"])

        return vs_train_op

    def construct_model(self):
        # initialize the placeholder
        self.visual_feat_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim, 2 * self.context_num + 1))
        self.sent_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.sentence_embedding_size))
        self.offset = tf.placeholder(tf.float32, shape=(self.batch_size,2))
        self.visual_feat_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim, 2 * self.context_num + 1))
        self.sent_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.sentence_embedding_size))
        # build inference network
        sim_reg_mat = self.visual_semantic_train(self.visual_feat_train, self.sent_train)
        sim_reg_mat_test = self.visual_semantic_test(self.visual_feat_test, self.sent_test)
        # compute loss
        self.loss_align_reg, offset_pred, loss_reg = self.compute_loss_reg(sim_reg_mat, self.offset)
        # optimize
        self.vs_train_op = self.training(self.loss_align_reg)

        return self.loss_align_reg, self.vs_train_op, sim_reg_mat_test, offset_pred, loss_reg

    def construct_test_model(self):
        # initialize the placeholder
        self.visual_feat_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim, 2 * self.context_num + 1))
        self.sent_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.sentence_embedding_size))
        # build inference network
        sim_reg_mat_test = self.visual_semantic_test(self.visual_feat_test, self.sent_test)

        return sim_reg_mat_test
