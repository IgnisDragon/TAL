import os
import tensorflow.compat.v1 as tf

#tf.disable_v2_behavior()

def save_ckpt(saver, sess, name=None, step=None, ckpt_dir='_checkpoints/'):

    if not os.path.exists(ckpt_dir): 
        os.makedirs(ckpt_dir)

    ckpt = os.path.join(ckpt_dir, name).replace('\\', '/')

    saver.save(sess, ckpt, global_step=step)
    print("[*] Saving model")
    
def load_ckpt(saver, sess, name=None, step=None, ckpt_dir='_checkpoints/'):

    if name:

        files = [x.split('.')[0] for x in os.listdir(ckpt_dir) if name in x and x.endswith('.index')]
        files.sort(key=lambda x: int(x.split('-')[1]))

        if len(files) == 0: return False, 0 

        if step:
            full_path = os.path.join(ckpt_dir, name + '-' + str(step))
            saver.restore(sess, full_path)
            return True, int(step)
        
        full_path = os.path.join(ckpt_dir, files[-1])     
        global_step = int(full_path.split('/')[-1].split('-')[-1])
        saver.restore(sess, full_path)

        return True, global_step
    
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