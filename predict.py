import tensorflow.compat.v1 as tf
import numpy as np
import time
from datetime import datetime
import operator

from dataset import movieDataset
import skip_thoughts.skipthoughts as skipthoughts
import ctrl_model
import utils.tf_saver as ckpt

tf.disable_v2_behavior()

def nms_temporal(x1, x2, s, overlap, top_n):
    pick = []
    assert len(x1) == len(s)
    assert len(x2) == len(s)
    if len(x1) == 0: return pick

    union = list(map(operator.sub, x2, x1)) # union = end - start
    # sort and get index of alignment score
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])]
    idx = [i for i in I if s[i] >= 0]

    while len(idx) > 0:
        i = idx[-1]
        pick.append(i)
        if len(pick) >= top_n: break

        xx1 = [max(x1[i], x1[j]) for j in idx[:-1]] # start
        xx2 = [min(x2[i], x2[j]) for j in idx[:-1]] # end
        inter = [max(0.0, k2 - k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u] / (union[i] + union[idx[u]] - inter[u]) for u in range(len(idx) - 1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <= overlap:
                I_new.append(idx[j])
        idx = I_new

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
        picks = nms_temporal(starts, ends, sim_v, iou_thresh-0.05, top_n)
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        #if top_n < len(picks): picks = picks[0:top_n]
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

    for query in query_feature:
        sent_vec = query[1]
        sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])
        print('----------------------------------------------------')
        test_result.write('----------------------------------------------------\n')
        results = []
        total_time = time.time()
        for k in range(len(visual_feature.movie_names)):
            start_time = time.time()
            movie_name = visual_feature.movie_names[k]
            print("loading " + movie_name, end=' ')
            movie_clip_featmaps = visual_feature.load_movie_slidingclip(movie_name)
            #print("clips: " + str(len(movie_clip_featmaps)))
            sentence_image_mat = np.zeros(len(movie_clip_featmaps))
            sentence_image_reg_mat = np.zeros([len(movie_clip_featmaps), 2])
            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0] # s13-d21_1_65.npy
                start = float(visual_clip_name.split("_")[1])
                end = float(visual_clip_name.split("_")[2].split(".")[0])
                #print(visual_clip_name + " " + str(start) + " " + str(end))
                featmap = np.reshape(featmap, [1, featmap.shape[0], 2 * model.context_num + 1])
                feed_dict = {
                    model.visual_feat_test: featmap,
                    model.sent_test:sent_vec
                }
                outputs = sess.run(vs_eval_op, feed_dict=feed_dict)
                sentence_image_mat[t] = outputs[0] # alignment score

                reg_end = end + outputs[2]
                reg_start = start + outputs[1]
                
                sentence_image_reg_mat[t, 0] = reg_start
                sentence_image_reg_mat[t, 1] = reg_end
            
            recall_top_n = recall_top_n_reg(top_n, IoU, sentence_image_mat, sentence_image_reg_mat, movie_name)
            for recall in recall_top_n: results.append(recall)

            duration = time.time() - start_time
            print('time: {:.2f} search...{:.1f}%'.format(duration, (k + 1) / len(visual_feature.movie_names) * 100.0))
            test_result.write('loading {} number of clips: {} time: {:.2f} search...{:.1f}%\n'
                        .format(movie_name, len(movie_clip_featmaps), duration, (k + 1) / len(visual_feature.movie_names) * 100.0))

        # remove duplicate in list
        ranks_set = set(tuple(x) for x in results)
        ranks = [list(x) for x in ranks_set]
        top_n_rank = [i for i in sorted(ranks, key=lambda x:x[0], reverse=True)]

        total_duration = time.time() - total_time
        print("\nQuery:{}\nIoU={}, R@{}:".format(query[0], IoU, top_n))    
        test_result.write("\nQuery:{}\nIoU={}, R@{}:\n".format(query[0], IoU, top_n))
        for i in range(len(top_n_rank)):
            if i < top_n: 
                print('{}. {} start: {:.0f} end: {:.0f} score: {:.2f}'
                        .format(i + 1, top_n_rank[i][1], top_n_rank[i][2], top_n_rank[i][3], top_n_rank[i][0]))
            if top_n_rank[i][0] > 0:
                test_result.write('{}. {} start: {:.0f} end: {:.0f} score: {:.2f}\n'
                            .format(i + 1, top_n_rank[i][1], top_n_rank[i][2], top_n_rank[i][3], top_n_rank[i][0]))
        
        print('- total time: {:.3f} average time: {:.3f}'.format(total_duration, total_duration / len(visual_feature.movie_names)))
        test_result.write('- total time: {:.3f} - average time: {:.3f}'.format(total_duration, total_duration / len(visual_feature.movie_names)))

def queryEncoder(query_sent_path):

    query_sent = []
    with open(query_sent_path) as f:
        for query in f:
            if query[0] != '_': query_sent.append(query)
        
    encoder = skipthoughts.Encoder(skipthoughts.load_model())
    query_vec = encoder.encode(query_sent)

    query_embed = []
    for i in range(len(query_vec)):
        query_embed.append([query_sent[i], query_vec[i]])

    print("query number: " + str(len(query_embed)))

    return query_embed

def run_predict(Iou=0.5, top_n=10):

    test_feature_dir = "E:/File/VS Code/DataSet/TACoS/Interval64_128_i3d_mixed5c/"
    query_file = "./dataset/query.txt"

    log = '{0:%Y-%m-%d-%H-%M}'.format(datetime.now())
    result_log = open("./results/{}_results.txt".format(log), "w")  
    
    model = ctrl_model.CTRL_Model()
    visual_feat = movieDataset(test_feature_dir)
    query_feat = queryEncoder(query_file)
    
    with tf.Graph().as_default():
		
        vs_eval_op = model.construct_test_model()
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        #config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # Run the Op to initialize the variables.
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(max_to_keep=10)
        
        load_model_status, _ = ckpt.load_ckpt(saver, sess, name='model')
        if load_model_status:
            print("[*] Model restore success!")
        else:
            print("[*] Not find pretrained model!")

        query_slidingclips(sess, vs_eval_op, model, visual_feat, query_feat, result_log, Iou, top_n)
    
    result_log.close()
    
if __name__ == '__main__':

    run_predict(Iou=0.1)
