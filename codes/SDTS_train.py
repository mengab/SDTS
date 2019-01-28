import tensorflow as tf
import h5py
import time
import flow
import numpy as np
import os
import random
import tflearn
import net
import random
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('img_width', 128,'input image width.')
tf.app.flags.DEFINE_integer('img_height',128,'input image width.')

tf.app.flags.DEFINE_integer('img_channel', 1,'number of image channel.')

tf.app.flags.DEFINE_integer('batch_size', 8,'batch size for training.')

tf.app.flags.DEFINE_integer('seq_length', 3,'total input and output length.')

tf.app.flags.DEFINE_integer('max_epochs', 30,'max num of steps.')

def train_batch_h5(f, length_f,start,batch_size):
    # rand_list = np.sort(random.sample(range(1,length_f),8))
    start_rand = random.randint(0,128)
    if start+batch_size > length_f-1:
        start =0
    image_pre_batch   = f.get('data_pre')[start:start+batch_size,...,start_rand:start_rand+128,start_rand:start_rand+128].transpose(0,3,2,1)
    image_cur_batch   = f.get('data_cur')[start:start+batch_size,...,start_rand:start_rand+128,start_rand:start_rand+128].transpose(0,3,2,1)
    image_aft_batch   = f.get('data_aft')[start:start+batch_size,...,start_rand:start_rand+128,start_rand:start_rand+128].transpose(0,3,2,1)
    image_label_batch = f.get('label')[start:start+batch_size,...,start_rand:start_rand+128,start_rand:start_rand+128].transpose(0,3,2,1)
    start = start+batch_size
    return image_pre_batch,image_cur_batch, image_aft_batch,image_label_batch,start


# def get_num_params():
#     num_params = 0
#     for variable in tf.trainable_variables():
#         shape = variable.get_shape()
#         num_params += reduce(mul, [dim.value for dim in shape], 1)
#     return num_params

def train():
    path1 = "../../dataset_256/train_256_b8_LD37_HP1.h5"
    path2 = "../../dataset_256/train_256_b8_LD37_HA1.h5"
    path3 = "../../dataset_256/train_256_b8_LD37_H1.h5"
    f1 = h5py.File(path1, 'r')
    f2 = h5py.File(path2, 'r')
    f3 = h5py.File(path3, 'r')
    train_net = net.Conv_Net_Train()
    length_f = min(f1['data_cur'].shape[0],f2['data_cur'].shape[0],f3['data_cur'].shape[0])

    lr = tf.placeholder(tf.float32,[])

    x1 = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_channel])
    x2 = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_channel])
    x3 = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_channel])
    x2_label = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_channel])
    
    ## MC-subnet
    x1to2,flow1To2 = flow.warp_img(FLAGS.batch_size, x2, x1, False)
    x3to2,flow3To2 = flow.warp_img(FLAGS.batch_size, x2, x3, True)

    enhanced_image =train_net.enhanced_Net(x1to2,x2,x3to2,is_train=True,name='enhanced_Net')

    l2_loss_1 = tf.nn.l2_loss(x1to2 - x2)
    l2_loss_2 = tf.nn.l2_loss(x3to2 - x2)

    l2_loss_3 = tf.nn.l2_loss(enhanced_image-x2_label)
    a=0.01
    b=1
    loss_total = a*(l2_loss_1+l2_loss_2 ) + b*(l2_loss_3) 

    configProt = tf.ConfigProto()
    configProt.gpu_options.allow_growth = True
    configProt.allow_soft_placement = True

    sess = tf.Session(config=configProt)

    optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
    train_op = optimizer.minimize(loss_total)
    init = tf.global_variables_initializer()
    sess.run(init)

    var = tf.trainable_variables()
    variables_to_restore = [val for val in var if 'easyflow' in val.name ]
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, './pre_flow_model/mdoel-ckpt-15')
    print('load successfully')
    lr_value = 0.0001
    saver = tf.train.Saver(var_list=tf.trainable_variables(),max_to_keep=60)
    # print('params:',get_num_params())
    for epoch in range(1, FLAGS.max_epochs+1):
        if epoch % 10 == 0:
            lr_value = lr_value * 0.1
        start1 = 0
        start2 = 0
        start3 = 0
        for itr in range((length_f*3)//8):
            time_start = time.time()
            if itr<=(length_f//8):
                image_pre_batch,image_cur_batch,image_aft_batch,label_batch, start1 = train_batch_h5(f1, length_f, start1, batch_size=FLAGS.batch_size)
            elif itr>(length_f//8) and itr<= (length_f*2)//8:
                image_pre_batch,image_cur_batch,image_aft_batch,label_batch, start2 = train_batch_h5(f2, length_f, start2, batch_size=FLAGS.batch_size)
            else:
                image_pre_batch,image_cur_batch,image_aft_batch,label_batch, start3 = train_batch_h5(f3, length_f, start3, batch_size=FLAGS.batch_size)

            feed_dict = {x1:image_pre_batch,x2:image_cur_batch,x3:image_aft_batch,x2_label:label_batch, lr: lr_value}

            # _, l2_loss_value, MC_image, lr_value_net = sess.run([train_op, loss_total, x1to2, lr],feed_dict)
            _, l2_loss_MC,l2_loss_3_net ,MC_image,enhanced, lr_value_net = sess.run([train_op, l2_loss_1,l2_loss_3, x1to2,enhanced_image, lr],feed_dict)
            time_end = time.time()
            time_step = time_end - time_start
            lr_value_net=np.mean(lr_value_net)
            if itr % 10 == 0:
                l1_loss_value = np.mean(np.abs((enhanced) - (label_batch)))
    
                total_time = time_step*((length_f*3)//8)*(FLAGS.max_epochs+1-epoch)/3600
                # print('itr:%d l1_loss:%f  lr:%f time_step:%f  total_time:%f' % (itr, l1_loss_value * 255.0, lr_value_net, time_step, time_step))
                print("===> Epoch[{}]({}/{}): lr:{:.10f} Loss_l1: {:.04f}: time_step:{:.04f} total_time:{:.04f}".format(epoch, itr, (length_f*3)//8,lr_value_net, l1_loss_value * 255.0,time_step,total_time))
                # print('===>Epoch[]({:.04f}/{:.04f})  '.format(l2_loss_MC, l2_loss_3_net))
        checkpoint_path='./266_SDTS/'
        os.makedirs("./266_SDTS/",exist_ok=True)
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.save(sess, checkpoint_path+'model-ckpt', global_step=epoch)
      
if __name__=='__main__':
    train()

