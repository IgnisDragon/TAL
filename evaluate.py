import numpy as np
import time
import datetime
import operator
import tensorflow.compat.v1 as tf
from sklearn.metrics import average_precision_score

import ctrl_model
from dataset import TestingDataSet
from dataset import TrainingDataSet
import utils.tf_saver as ckpt

tf.logging.set_verbosity(tf.logging.WARN) # DEBUG, INFO, WARN, ERROR
tf.disable_v2_behavior()

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def compute_ap(class_score_matrix, labels):
    num_classes=class_score_matrix.shape[1]
    one_hot_labels=dense_to_one_hot(labels, num_classes)
    predictions=np.array(class_score_matrix>0, dtype="int32")
    average_precision=[]
    for i in range(num_classes):
        ps=average_precision_score(one_hot_labels[:, i], class_score_matrix[:, i])
       # if not np.isnan(ps):
        average_precision.append(ps)
    return np.array(average_precision)

def calculate_IoU(i0,i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return iou

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
def compute_IoU_recall_top_n(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):

    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]): # len(movie_clip_sentences)
        gt = sclips[k] # ex: s13-d21_1_65
        gt_start = float(gt.split("_")[1]) # 1
        gt_end = float(gt.split("_")[2]) # 65
        # print(gt + " " + str(gt_start) + " " + str(gt_end))
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k,:,0]]
        ends = [e for e in sentence_image_reg_mat[k,:,1]]
        picks = nms_temporal(starts, ends, sim_v, iou_thresh-0.05, top_n)
        #if top_n < len(picks): picks = picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if iou >= iou_thresh:
                correct_num+=1
                break
    return correct_num

'''
evaluate the model
'''
def do_eval_slidingclips(sess, vs_eval_op, model, test_set, iter_step, log_time):
    IoU_thresh = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_correct_num_10 = [0.0] * 5
    all_correct_num_5 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0
    for movie_name in test_set.movie_names:
        eval_time = time.time()
        print("Test movie: " + movie_name + "....loading movie data")
        movie_clip_featmaps, movie_clip_sentences = test_set.load_movie_slidingclip(movie_name, 16)
        print("sentences: " + str(len(movie_clip_sentences)))
        print("clips: " + str(len(movie_clip_featmaps)))
        sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])

        for k in range(len(movie_clip_sentences)):
            sent_vec = movie_clip_sentences[k][1]
            sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])

            for t in range(len(movie_clip_featmaps)):

                visual_clip_name = movie_clip_featmaps[t][0] # s13-d21_1_65.npy
                start = float(visual_clip_name.split("_")[1]) # 1
                end = float(visual_clip_name.split("_")[2].split(".")[0]) # 65
                #print(visual_clip_name + " " + str(start) + " " + str(end))
                featmap = movie_clip_featmaps[t][1] # 4096, 3
                featmap = np.reshape(featmap, [1, featmap.shape[0], 2 * model.context_num + 1])
                
                feed_dict = {
                    model.visual_feat_test: featmap,
                    model.sent_test:sent_vec
                }
                outputs = sess.run(vs_eval_op, feed_dict=feed_dict)

                sentence_image_mat[k,t] = outputs[0] # alignment score
                #reg_clip_length = (end - start) * (10**outputs[2])
                #reg_mid_point = (start + end) / 2.0 + movie_length * outputs[1]
                sentence_image_reg_mat[k,t,0] = start + outputs[1]
                sentence_image_reg_mat[k,t,1] = end + outputs[2]
        
        iclips = [b[0] for b in movie_clip_featmaps] # clip_names, comb_feature
        sclips = [b[0] for b in movie_clip_sentences] # clip_names, sent_vec
        
        print("evalute....")
        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            print('{} IoU={}, R@10: {:.3f}; R@5: {:.3f}; R@1: {:.3f}'
                .format(movie_name, IoU, correct_num_10 / len(sclips), 
                                        correct_num_5 / len(sclips), 
                                        correct_num_1 / len(sclips)))
            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1

        all_retrievd += len(sclips)
        duration = time.time() - eval_time
        print('- time: {}'.format(datetime.timedelta(seconds=duration)))

    print('---------------results-----------------')
    eval_output = open("./results/{}_eval.txt".format(log_time), "a")
    for k in range(len(IoU_thresh)):
        if (k == 0): eval_output.write("Step " + str(iter_step) + '\n')
            
        print('IoU={}, R@10: {:.3f}; R@5: {:.3f}; R@1: {:.3f}'
            .format(IoU_thresh[k], all_correct_num_10[k] / all_retrievd, 
                                    all_correct_num_5[k] / all_retrievd, 
                                    all_correct_num_1[k] / all_retrievd))
        eval_output.write('IoU={}, R@10: {:.3f}; R@5: {:.3f}; R@1: {:.3f}\n'
            .format(IoU_thresh[k], all_correct_num_10[k] / all_retrievd,
                                    all_correct_num_5[k] / all_retrievd, 
                                    all_correct_num_1[k] / all_retrievd))

    eval_output.close()

def run_training(max_steps=20000, batch_size=50, context_size=1, load_model=True):

    train_feature_dir = "E:/File/VS Code/DataSet/TACOS/Interval64_128_i3d_mixed5c/"
    test_feature_dir = "E:/File/VS Code/DataSet/TACoS/Interval128_i3d_mixed5c/"
    train_csv_path = "./exp_data/train_clip_sentence.pkl"
    test_csv_path = "./exp_data/test_clip_sentence.pkl"
    log_time = '{0:%Y-%m-%d-%H-%M}'.format(datetime.datetime.now())

    train_set = TrainingDataSet(train_feature_dir, train_csv_path, batch_size)
    test_set = TestingDataSet(test_feature_dir, test_csv_path, 1)
    model = ctrl_model.CTRL_Model(batch_size, context_size)

    training_time = time.time()
    with tf.Graph().as_default():
		
        loss_align_reg, train_op, eval_op, offset_pred, loss_reg, loss_align = model.construct_model()
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        config = tf.ConfigProto(gpu_options = gpu_options)
        #config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # Run the Op to initialize the variables.
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(max_to_keep=10)
        if load_model:
            load_model_status, global_step = ckpt.load_ckpt(saver, sess, name='model')
            if load_model_status:
                print("[*] Model restore success!")
            else:
                print("[*] Not find pretrained model!")

        for step in range(global_step, max_steps):
            start_time = time.time()

            image_batch, sentence_batch, offset_batch = train_set.next_batch_iou()
            feed_dict = model.fill_feed_dict_train_reg(image_batch, sentence_batch, offset_batch)
            _, loss, loss_r, loss_a, offsets = sess.run([train_op, loss_align_reg, loss_reg, loss_align, offset_pred], 
                                                        feed_dict=feed_dict)

            if (step + 1) % 50 == 0:
                print('Step {}: loss = {:.3f}, reg = {:.3f}, align = {:.3f} ({:.3f} sec)'
                        .format(step + 1, loss, loss_r, loss_a, time.time() - start_time))
                
                #print('predict:')
                #for i in range(1):
                #    print('gt: {}, {} pred: {:.1f}, {:.1f}'
                #            .format(offset_batch[i][0], offset_batch[i][1], offsets[i][0], offsets[i][1]))
                
            if (step + 1) % 2000 == 0:
                ckpt.save_ckpt(saver, sess, name='model', step=step + 1)
                print("Start to test:-----------------")
                do_eval_slidingclips(sess, eval_op, model, test_set, step + 1, log_time)              
    
    duration = time.time() - training_time
    print('- total time: {}'.format(datetime.timedelta(seconds=duration)))

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
