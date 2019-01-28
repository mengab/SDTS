import tensorflow as tf
import numpy as np
import time
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
# import tflearn

class Conv_Net_Train(object):
    def __init__(self,is_training = False,batch_normalization = False):
        if batch_normalization:
            self.normalizer = slim.batch_norm
            self.is_training = is_training
        else:
            self.normalizer = None
            self.is_training = False
        self.norm_params1 = {'is_training': self.is_training, 'decay': 0.9,
                        'epsilon': 1e-5, 'updates_collections': None}


    def pre_convolution(self,image1,image2,image3,name):
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                # normalizer_fn=self.normalizer,
                                # normalizer_params=self.norm_params1,
                                weights_initializer=initializers.xavier_initializer(uniform=True),
                                weights_regularizer=slim.l1_regularizer(1e-4)
                                ):
                image1 = slim.conv2d(image1, 64, [3, 3], 1, scope='conv1_1')
                image1 = slim.conv2d(image1, 64, [3, 3], 1, scope='conv1_2')
                image2 = slim.conv2d(image2, 64, [3, 3], 1, scope='conv2_1')
                image2 = slim.conv2d(image2, 64, [3, 3], 1, scope='conv2_2')
                image3 = slim.conv2d(image3, 64, [3, 3], 1, scope='conv3_1')
                image3 = slim.conv2d(image3, 64, [3, 3], 1, scope='conv3_2')

                image_1_2 = tf.concat([image1,image2],axis =3)
                image_1_2 = slim.conv2d(image_1_2, 64, [3, 3], 1, scope='conv_1_2')

                image_2_3 = tf.concat([image2,image3],axis =3)
                image_2_3 = slim.conv2d(image_2_3, 64, [3, 3], 1, scope='conv_2_3')

                image_1_2_3 = tf.concat([image_1_2,image_2_3],axis =3)
                image_1_2_3 = slim.conv2d(image_1_2_3, 64, [3, 3], 1, scope='conv_1_2_3')
                return image_1_2_3

    def res_block(self,input,name):
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                # normalizer_fn=self.normalizer,
                                # normalizer_params=self.norm_params1,
                                weights_initializer=initializers.xavier_initializer(uniform=True),
                                weights_regularizer=slim.l1_regularizer(1e-4)
                                ):
                # print('......................................')
                split1 = input
                split1_1 = input
                conv3_1 = slim.conv2d(split1,48,[3,3],1,scope='conv_3_1')
                conv3_2 = slim.conv2d(conv3_1,48,[3,3],1,scope='conv3_2')
                slice1_1,slice1_2 = tf.split(conv3_2,[16,32],axis=3)
                conv3_3 = slim.conv2d(slice1_2,48,[3,3],scope='conv3_3')
                conv3_4 =slim.conv2d(conv3_3,64,[3,3],scope='conv3_4')
                slice2_1,slice2_2 = tf.split(conv3_4,[16,48],axis=3)
                conv3_5 = slim.conv2d(slice2_2,48,[3,3],1,scope='conv3_5')
                conv3_6 = slim.conv2d(conv3_5,96,[3,3],1,scope='conv3_6')

                concat1 = tf.concat([split1_1,slice1_1,slice2_1],axis=3)

                sum1 = concat1 + conv3_6
                down1 = slim.conv2d(sum1,64,[1,1],1,scope='down1')
                return down1

   
    def enhanced_Net(self,image1,image2,image3,is_train=True,name='enhanced_Net'):
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                # normalizer_fn=self.normalizer,
                                # normalizer_params=self.norm_params1,
                                weights_initializer=initializers.xavier_initializer(uniform=True),
                                weights_regularizer=slim.l1_regularizer(1e-4)
                                ):

                image1_2_3 = self.pre_convolution(image1,image2,image3,name=name)
                
                conv2=slim.conv2d(image1_2_3,64,[3,3],1,scope='conv2')

                down1 = self.res_block(input=conv2,name='conv3')

                down2 = self.res_block(input = down1,name='conv4')

                down3 = self.res_block(input=down2,name='conv5')

                # down4 = self.res_block(input = down3,name = 'conv6')

                conv7 = slim.conv2d(down3,64,[3,3],1,scope='conv7')
                conv8 = slim.conv2d(conv7,1,[3,3],1,activation_fn=None,scope='conv8')
                if is_train:
                    conv8_out =conv8+image2
                else:
                    conv8_out =conv8+image2
                    conv8_out = tf.clip_by_value(conv8_out ,0,1.0)
                return conv8_out

   