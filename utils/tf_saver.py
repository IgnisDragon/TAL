import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

def save_ckpt(sess, iter_num, ckpt_dir='_checkpoints/'):

    if not os.path.exists(ckpt_dir): 
        os.makedirs(ckpt_dir)

    saver = tf.train.Saver()
    saver.save(sess, ckpt_dir, global_step=iter_num)
    print("[*] Saving model")
    
def load_ckpt(sess, ckpt_dir='_checkpoints/'):

    if not os.path.exists(ckpt_dir): 
        os.makedirs(ckpt_dir)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    if ckpt and ckpt.model_checkpoint_path:

        full_path = tf.train.latest_checkpoint(ckpt_dir)

        try:
            global_step = int(full_path.split('/')[-1].split('-')[-1])

        except ValueError:
            global_step = None

        saver.restore(sess, full_path)

        return True, global_step

    else:
        print("[*] Failed to load model from %s" % ckpt_dir)      
        return False, 0    

'''
example:

load_model_status, global_step = load_ckpt(sess, ckpt_dir)

if load_model_status:
    print("[*] Model restore success!")
else:
    sess.run(tf.global_variables_initializer())
    print("[*] Not find pretrained model!")

print("[*] Start training step %d : " % (iter_num))

save_ckpt(sess, iter_num + 1, ckpt_dir)
'''