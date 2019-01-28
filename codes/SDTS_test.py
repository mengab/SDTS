#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from scipy.misc import imresize
from skimage.measure import compare_ssim
import cv2
import glob
import os
import argparse
import numpy as np
import flow
import tensorflow as tf
import math
import net

import copy
import time

def yuv_import_8_bits(filename, dims ,startfrm,numframe):
    fp = open(filename, 'rb')
    frame_size = np.prod(dims) * 3 / 2
    fp.seek(0, 2)
    ps = fp.tell()
    totalfrm = int(ps // frame_size) 
    d00 = dims[0] // 2
    d01 = dims[1] // 2
    assert startfrm+numframe<=totalfrm
    Y = np.zeros(shape=(numframe, 1,dims[0], dims[1]), dtype=np.uint8, order='C')
    U = np.zeros(shape=(numframe, 1,d00, d01),dtype= np.uint8, order='C')
    V = np.zeros(shape=(numframe, 1,d00, d01),dtype= np.uint8, order='C')

    fp.seek(int(frame_size * startfrm), 0)
    for i in range(startfrm,startfrm+numframe):
        for m in range(dims[0]):
            for n in range(dims[1]):
                pel8bit = int.from_bytes(fp.read(1), byteorder='little',signed=False)
                Y[i-startfrm,0, m, n] = np.uint8(pel8bit)
        for m in range(d00):
            for n in range(d01):
                pel8bit = int.from_bytes(fp.read(1), byteorder='little',signed=False)
                U[i-startfrm,0, m, n] = np.uint8(pel8bit)                
        for m in range(d00):
            for n in range(d01):
                pel8bit = int.from_bytes(fp.read(1), byteorder='little',signed=False)
                V[i-startfrm,0, m, n] = np.uint8(pel8bit)
                
    fp.close()
    Y = Y.astype(np.float32)
    U = U.astype(np.float32)
    V = V.astype(np.float32)
    return Y, U, V

def yuv_import_10bits(filename, dims ,startfrm,numframe):
    fp = open(filename, 'rb')
    frame_size = np.prod(dims) * 3 
    
    d00 = dims[0] // 2
    d01 = dims[1] // 2
    #assert startfrm+numframe<=totalfrm
    Y = np.zeros(shape=(numframe, 1,dims[0], dims[1]), dtype=np.uint16, order='C')
    U = np.zeros(shape=(numframe, 1,d00, d01),dtype= np.uint16, order='C')
    V = np.zeros(shape=(numframe, 1,d00, d01),dtype= np.uint16, order='C')
    fp.seek(int(frame_size * startfrm), 0)
    for i in range(startfrm,startfrm+numframe):
        for m in range(dims[0]):
            for n in range(dims[1]):
                pel10bit =  int.from_bytes(fp.read(2), byteorder='little', signed=False)
                Y[i-startfrm,0, m, n] = pel10bit 
                
        for m in range(d00):
            for n in range(d01):
                pel10bit_u = int.from_bytes(fp.read(2), byteorder='little', signed=False)
                U[i-startfrm,0, m, n]= pel10bit_u 
                
        for m in range(d00):
            for n in range(d01):
                pel10bit_v = int.from_bytes(fp.read(2), byteorder='little', signed=False)
                V[i-startfrm,0, m, n]  = pel10bit_v 


    fp.close()
    Y = Y.astype(np.float32)
    U = U.astype(np.float32)
    V = V.astype(np.float32)
    return Y, U, V



def get_w_h(filename):
    width = int((filename.split('x')[0]).split('_')[-1])
    height = int((filename.split('x')[1]).split('_')[0])
    return (height, width)

def get_data(one_filename,video_index,num_frame):
    
    one_filename_length = len(one_filename)
    data_Y = []
    for i in range(one_filename_length+1):
        if i == 0:
            data_37_filename = np.sort(glob.glob(one_filename[i]+'/*.yuv'))
            # print(data_37_filename)
            data_37_filename_length = len(data_37_filename )
            for i_0 in range(video_index,video_index+1):
                file_name = data_37_filename[i_0]
                dims = get_w_h(filename=file_name)
                data_37_filename_Y,data_37_filename_U,data_37_filename_V = yuv_import_10bits(filename=file_name, dims=dims ,startfrm=4,numframe=num_frame)
                data_Y.append(data_37_filename_Y)
                
        if i == 1:
            label_37_filename = np.sort(glob.glob('../test_yuv/label/' + '*.yuv'))
            
            label_37_filename_length = len(label_37_filename)
           
            for i_2 in range(video_index,video_index+1):
                file_name = label_37_filename[i_2]
                dims = get_w_h(filename=file_name)
                label_37_filename_Y, label_37_filename_U, label_37_filename_V = yuv_import_8_bits(filename=file_name, dims=dims,startfrm=4, numframe=num_frame)
                data_Y.append(label_37_filename_Y)
              

    return  data_Y


def test_batch(data_Y, start, batch_size=1):
    pre = (start//4)*4
    aft = (start//4)*4+4
    data_pre = (data_Y[0][pre:pre+1,...])/1023.
    data_cur = data_Y[0][start:start+1,...]/1023.
    data_aft = data_Y[0][aft:aft+1,...]/1023.

    label    = data_Y[1][start:start+1,...]

    start+=1
    return  data_pre,data_cur,data_aft,label,start


def PSNR(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2).astype(np.float32)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def patch_test(one_filename, net_G,warp_img_1to2,warp_img_3to2, sess,x1,x2,x3, patch_size, f_txt):
    video_num=1
    ave_diff_psnr    =0.
    ave_diff_ssim    =0.
    ave_psnr_pre_gt  =0.
    ave_psnr_data_gt =0.
    ave_ssim_pre_gt  =0.
    ave_ssim_data_gt =0.
    
    for video_index in range(video_num):
        data_37_filename = np.sort(glob.glob(one_filename[0]+'/*.yuv'))
        data_Y = get_data(one_filename,video_index=video_index,num_frame=25)
        start =1
        patch_size = patch_size
        psnr_diff_sum = 0
        psnr_pre_gt_sum=0
        psnr_data_gt_sum=0

        ssim_diff_sum = 0
        ssim_pre_gt_sum=0
        ssim_data_gt_sum =0
        nums_every_video =20
        nums=0
        for itr in range(0, nums_every_video):
            if (start%4)!=0:
                nums+=1
                data_pre, data_cur, data_aft, label, start = test_batch(data_Y=data_Y, start=start, batch_size=1)
                    
                height = data_pre.shape[2]
                width = data_pre.shape[3]
                # print(height,width)
                image_synthesis = np.zeros((height , width ), np.float32)
                image_mask = np.zeros((height , width ), np.float32)

                num_rows = height // patch_size
                num_cols = width // patch_size
                row_redundant = int(height % patch_size)
                col_redundant = int(width % patch_size)
                result_row_redundant = int((height ) % (patch_size ))
                result_col_redundant = int((width ) % (patch_size ))
                row_start = 0
                result_row_start = 0
                patch_step = 0
                for row in range(num_rows + 1):
                    col_start = 0
                    result_col_start = 0
                    for col in range(num_cols + 1):
                        patch_step += 1
                        data_pre_value_patch = data_pre[:, :,  row_start:row_start + patch_size,col_start:col_start + patch_size]

                        data_cur_value_patch = data_cur[:, :,  row_start:row_start + patch_size,col_start:col_start + patch_size]

                        data_aft_value_patch = data_aft[:, :,  row_start:row_start + patch_size,col_start:col_start + patch_size]
                        
                        data_pre_value_patch = data_pre_value_patch.transpose(0, 2, 3, 1)
                        data_cur_value_patch = data_cur_value_patch.transpose(0, 2, 3, 1)
                        data_aft_value_patch = data_aft_value_patch.transpose(0, 2, 3, 1)
                        start_time=time.time()
                        fake_image = sess.run(net_G,feed_dict={x1:data_pre_value_patch ,x2:data_cur_value_patch,x3:data_aft_value_patch})
                        end_time=time.time()
                        if  patch_step==3 and itr==2:
                            print('192x192 time:%.04f(ms):'%(end_time-start_time))
                        
                        fake_image_numpy = np.array(fake_image)
                        fake_image_numpy = np.squeeze(fake_image_numpy)
                        # print('fake_image_numpy:',fake_image_numpy.shape)
                        image_synthesis[result_row_start:result_row_start + patch_size ,result_col_start:result_col_start + patch_size ] += copy.copy(np.array(fake_image_numpy))
                        image_mask[result_row_start:result_row_start + patch_size ,result_col_start:result_col_start + patch_size ] += 1.0
                        if col == 0:
                            col_start = col_start + col_redundant
                            result_col_start = result_col_start + result_col_redundant
                        else:
                            col_start += patch_size
                            result_col_start += patch_size
                      
                    if row == 0:
                        row_start += row_redundant
                        result_row_start = result_row_start + result_row_redundant
                    else:
                        row_start += patch_size
                        result_row_start += patch_size

                finally_image = ((((image_synthesis / image_mask)*255.0))).astype(np.float32)
                finally_image=np.squeeze(finally_image)
               
                os.makedirs('./result/result/%02d'%(video_index+1),exist_ok = True)
                os.makedirs('./result/result_label/%02d'%(video_index+1),exist_ok = True)
                os.makedirs('./result/result_raw_data/%02d'%(video_index+1),exist_ok = True)
                cv2.imwrite('./result/result/%02d/%02d.png'%(video_index+1,itr+2),finally_image.astype(np.uint8))
                data_cur_image = (np.squeeze(data_cur)*255.0).astype(np.float32)
                label = np.squeeze(label).astype(np.float32)
                cv2.imwrite('./result/result_label/%02d/%02d.png'%(video_index+1,itr+2),label.astype(np.uint8))
                cv2.imwrite('./result/result_raw_data/%02d/%02d.png'%(video_index+1,itr+2),data_cur_image.astype(np.uint8))
                psnr_pre_gt  = PSNR(finally_image, label)

                psnr_data_gt = PSNR(data_cur_image, label)

                psnr_diff = psnr_pre_gt - psnr_data_gt
                psnr_diff_sum +=psnr_diff
                psnr_pre_gt_sum+=psnr_pre_gt
                psnr_data_gt_sum+=psnr_data_gt
                ssim_pre_gt = compare_ssim(finally_image.astype(np.uint8), label.astype(np.uint8))
                ssim_data_gt = compare_ssim(data_cur_image.astype(np.uint8), label.astype(np.uint8))
                ssim_diff = ssim_pre_gt - ssim_data_gt
                ssim_diff_sum += ssim_diff
                ssim_pre_gt_sum+=ssim_pre_gt
                ssim_data_gt_sum+=ssim_data_gt
                print('psnr_pre_gt:{:.04f} psnr_data_gt:{:.04f}  psnr_diff:{:.04f} ssim_pre_gt:{:.04f} ssim_data_gt:{:.04f}  ssim_diff:{:.04f}'.format(psnr_pre_gt,psnr_data_gt,psnr_diff,ssim_pre_gt,ssim_data_gt,ssim_diff),file=f_txt)
            else:
                start+=1
        print( data_37_filename[video_index],'----',"video_index:%02d,psnr_ave:%.04f,ssim_ave:%.04f"%(video_index,psnr_diff_sum/nums,ssim_diff_sum/nums))
        print(' video_index:{:2d} psnr_pre_gt_ave:{:.04f} psnr_data_gt_ave:{:.04f}  psnr_diff_ave:{:.04f} ssim_pre_gt_ave:{:.04f} ssim_data_gt_ave:{:.04f} ssim_diff_ave:{:.04f}'.format(video_index,psnr_pre_gt_sum/nums,psnr_data_gt_sum/nums,psnr_diff_sum/nums,ssim_pre_gt_sum/nums,ssim_data_gt_sum/nums,ssim_diff_sum/nums),file=f_txt)
        print('{}'.format(data_37_filename[video_index]),file=f_txt)
        f_txt.write('\r\n')
        ave_diff_psnr+=psnr_diff_sum/nums
        ave_diff_ssim+=ssim_diff_sum/nums
        ave_psnr_pre_gt  +=psnr_pre_gt_sum/nums
        ave_psnr_data_gt +=psnr_data_gt_sum/nums
        ave_ssim_pre_gt  +=ssim_pre_gt_sum/nums
        ave_ssim_data_gt +=ssim_data_gt_sum/nums
    print(' ave_psnr_pre_gt:{:.04f} ave_psnr_data_gt:{:.04f}  psnr_diff_ave:{:0.4f} ave_ssim_pre_gt:{:.04f} ave_ssim_data_gt:{:.04f} ssim_diff_ave:{:.04f}'.format(ave_psnr_pre_gt/video_num,ave_psnr_data_gt/video_num,ave_diff_psnr/video_num,ave_ssim_pre_gt/video_num,ave_ssim_data_gt /video_num,ave_diff_ssim/video_num))
  
    print(' ave_psnr_pre_gt:{:.04f} ave_psnr_data_gt:{:.04f}  psnr_diff_ave:{:0.4f} ave_ssim_pre_gt:{:.04f} ave_ssim_data_gt:{:.04f} ssim_diff_ave:{:.04f}'.format(ave_psnr_pre_gt/video_num,ave_psnr_data_gt/video_num,ave_diff_psnr/video_num,ave_ssim_pre_gt/video_num,ave_ssim_data_gt /video_num,ave_diff_ssim/video_num), file=f_txt)




if __name__ == "__main__":
    num = 0
    parser = argparse.ArgumentParser(description="2019_test")
    train_net = net.Conv_Net_Train()
    opts = parser.parse_args()
    
    txt_name = './LD37_no_key.txt'
    if os.path.isfile(txt_name):
        f = open(txt_name, 'w+')  
    else:
        os.mknod(txt_name)
        f = open(txt_name, 'w+')

    one_filename = np.sort(glob.glob('../test_yuv/LD37/' + '*'))
    patch_size = 192
    Channel = 1
    batch_size = 1
    # Session
    configProt = tf.ConfigProto()
    configProt.gpu_options.allow_growth = True
    configProt.allow_soft_placement = True

    sess = tf.Session(config=configProt)

    # Placeholder
    x1 = tf.placeholder(tf.float32, [batch_size, patch_size,  patch_size, Channel])
    x2 = tf.placeholder(tf.float32, [batch_size, patch_size,  patch_size, Channel])
    x3 = tf.placeholder(tf.float32, [batch_size, patch_size,  patch_size, Channel])

    ## MC-subnet
    x1to2,flow1to2 = flow.warp_img(batch_size, x2, x1, False)
    x3to2,flow3to2 = flow.warp_img(batch_size, x2, x3, True)

    ## QE-subnet
    x2_enhanced = train_net.enhanced_Net(x1to2, x2, x3to2,is_train=False,name='enhanced_Net')
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    saver.restore(sess, '../model_LD37/model-ckpt-30')

    patch_test(one_filename=one_filename, net_G=x2_enhanced,warp_img_1to2=x1to2,warp_img_3to2=x3to2, sess=sess,x1=x1,x2=x2,x3=x3, patch_size=patch_size, f_txt=f)

    f.close()
 



