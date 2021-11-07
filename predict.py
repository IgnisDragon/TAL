import tensorflow.compat.v1 as tf
import numpy as np
import time
import operator

import dataset
import ctrl_model

from utils.tf_saver import *

tf.disable_v2_behavior()

def nms_temporal(x1, x2, s, overlap):
    pick = []
    assert len(x1) == len(s)
    assert len(x2) == len(s)
    if len(x1) == 0: return pick

    union = list(map(operator.sub, x2, x1)) # union = x2 - x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index of alignment score
    
    while len(I) > 0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i], x1[j]) for j in I[:-1]] # start >= start_ali(hige alignment score)
        xx2 = [min(x2[i], x2[j]) for j in I[:-1]] # end <= end_ali
        inter = [max(0.0, k2 - k1) for k1, k2 in zip(xx1, xx2)] # inter <= union
        o = [inter[u] / (union[i] + union[I[u]] - inter[u]) for u in range(len(I) - 1)] # Similarity
        I_new = []
        for j in range(len(o)): # if o[j] > overlap then remove, the lower overlap, the more similar clip
            if o[j] <= overlap:
                I_new.append(I[j])
        I = I_new
    
    return pick 

'''
compute recall at certain IoU
'''
def recall_top_n_reg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, movie_name):

    predict = []
    for k in range(sentence_image_mat.shape[0]):
        # print(gt +" "+str(gt_start)+" "+str(gt_end))
        sim_v = [v for v in sentence_image_mat]
        starts = [s for s in sentence_image_reg_mat[:,0]]
        ends = [e for e in sentence_image_reg_mat[:,1]]
        picks = nms_temporal(starts, ends, sim_v, iou_thresh)
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n < len(picks): picks = picks[0:top_n]
        for index in picks:
            pred_sim_v = sentence_image_mat[index]
            pred_start = int(sentence_image_reg_mat[index, 0])
            pred_end = int(sentence_image_reg_mat[index, 1])
            predict.append([pred_sim_v, movie_name, pred_start, pred_end])

    return predict

'''
query sentence from extract feature data
'''
def query_slidingclips(sess, vs_eval_op, model, visual_feature, query_feature, test_result, IoU=0.5, top_n=10):

    for query in query_feature.sent_feat:
        sent_vec = query[1]
        sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])

        total = []
        for k in range(len(visual_feature.movie_names)):
            start_time = time.time()
            movie_name = visual_feature.movie_names[k]
            print("loading " + movie_name, end=' ')
            movie_clip_featmaps = visual_feature.load_movie_slidingclip(movie_name)
            #print("clips: " + str(len(movie_clip_featmaps)))
            sentence_image_mat = np.zeros(len(movie_clip_featmaps))
            sentence_image_reg_mat = np.zeros([len(movie_clip_featmaps), 2])
            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1] # 4096 * 3
                visual_clip_name = movie_clip_featmaps[t][0]
                start = float(visual_clip_name.split("_")[1])
                end = float(visual_clip_name.split("_")[2].split("_")[0])
                #print(visual_clip_name + " " + str(start) + " " + str(end))
                featmap = np.reshape(featmap, [1, featmap.shape[0]])
                feed_dict = {
                    model.visual_featmap_ph_test: featmap,
                    model.sentence_ph_test:sent_vec
                }
                outputs = sess.run(vs_eval_op, feed_dict=feed_dict)
                sentence_image_mat[t] = outputs[0] # alignment score
                #reg_clip_length = (end - start) * (10**outputs[2])
                #reg_mid_point = (start + end) / 2.0 + movie_length * outputs[1]
                reg_end = end + outputs[2]
                reg_start = start + outputs[1]
                
                sentence_image_reg_mat[t, 0] = reg_start
                sentence_image_reg_mat[t, 1] = reg_end
            
            recall_top_n = recall_top_n_reg(top_n, IoU, sentence_image_mat, sentence_image_reg_mat, movie_name)
            for recall in recall_top_n: total.append(recall)

            duration = time.time() - start_time
            print('time: {:.2f} search...{:.1f}%'.format(duration, (k + 1) / len(visual_feature.movie_names) * 100.0))
            test_result.write('loading {} number of clips: {} time: {:.2f} search...{:.1f}%\n'
                        .format(movie_name, len(movie_clip_featmaps), duration, (k + 1) / len(visual_feature.movie_names) * 100.0))

        ranks_set = set(tuple(x) for x in total)
        ranks = [list(x) for x in ranks_set]
        top_n_rank = [i for i in sorted(ranks, key=lambda x:x[0], reverse=True)]

        print("IoU={}, R@{}:".format(IoU, top_n))
        test_result.write("Query: " + query[0] + "\n")
        test_result.write("IoU={}, R@{}:\n".format(IoU, top_n))
        for i in range(len(top_n_rank)):
            if i < top_n: 
                print('{}. {} start: {:.0f} end: {:.0f}'
                        .format(i + 1, top_n_rank[i][1], top_n_rank[i][2], top_n_rank[i][3]))
            if top_n_rank[i][0] > 0:
                test_result.write('{}. {} start: {:.0f} end: {:.0f} score: {:.2f}\n'
                            .format(i + 1, top_n_rank[i][1], top_n_rank[i][2], top_n_rank[i][3], top_n_rank[i][0]))

def run_predict():

    test_feature_dir = "E:/File/VS Code/DataSet/TACoS/Interval128_256_overlap0.8_c3d_fc6/"
    #train_feature_dir = "E:/File/VS Code/DataSet/TACOS/Interval64_128_256_512_overlap0.8_c3d_fc6/"
    query_file = "./dataset/query.txt"

    log = time.strftime("%Y-%m-%d %H:%M", time.localtime()) 
    result_log = open("./results/{}_results.txt".format(log), "w")  

    model = ctrl_model.CTRL_Model()
    visual_feat = dataset.movieDataset(test_feature_dir)
    query_feat = dataset.queryEncoder(query_file)

    with tf.Graph().as_default():
		
        vs_eval_op = model.construct_test_model()
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        config = tf.ConfigProto(gpu_options = gpu_options)
        #config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()
        sess.run(init)
        load_model_status, _ = load_ckpt(sess)

        if load_model_status:
            print("[*] Model restore success!")
            vars = tf.trainable_variables()
            for var in vars: print(var.name)
        else:
            print("[*] Not find pretrained model!")

        start_time = time.time()
        query_slidingclips(sess, vs_eval_op, model, visual_feat, query_feat, result_log)
        duration = time.time() - start_time
        print("times: " + str(duration))
    
    result_log.close()

if __name__ == '__main__':
    run_predict()
