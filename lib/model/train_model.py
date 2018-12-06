from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# this files are temporary and need update
import keras
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from keras.optimizers import Adam, Adadelta, SGD

from config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
from multiprocessing import Process, Lock, Pipe
from time import sleep

from nets.RPN import RPN
from nets.RegNet import RegNet
from nets.NetBuild import NetBuild
from loss.Losses import Losses
from datasets.DataGen import DataGen

# for debug
from utils.bbox_transform import clip_boxes, bbox_transform_inv
from utils.nms_wrapper import nms



def build_model(imdb, net, pre_net_path):
    # --------------------build the model-------------------

    num_classes = imdb.num_classes
    num_anchors = len(cfg.ANCHOR_SCALES)*len(cfg.ANCHOR_RATIOS)
    number_channel = 3

    input_shape = (None, None, number_channel)

    #train sets
    image_input = Input(shape=input_shape)    
    im_info_input = Input(shape=(3, ))
    gt_box_input = Input(shape=(None, 5))

    #test sets
    image_input_test = Input(shape=input_shape, name='image_input_test')
    im_info_input_test = Input(shape=(3, ), name='im_info_input_test')


    #VGG16(), RPN(), RegNet(), Losses()
    '''
    1. Conv Layer
    2. Region Proposal Network (RPN)
    3. Roi Pooling
    4. Classification 
    '''

    #--------- Conv Layer, output = feature maps
    # convolution feature
    model_feat_conv = net    
    model_feat_conv.conv_feature(input_shape, cfg.RPN_CHANNELS)
    model_feat_conv.load_weight(pre_net_path)

    feat_conv = model_feat_conv.model_seq(image_input)
    # testing model
    feat_conv_test = model_feat_conv.model_seq(image_input_test)

    model_feat_conv.model_seq.name="feat_conv"

    #-------- Region Proposal Network (RPN), output = proposals
    # RPN
    model_RPN = RPN()
    feat_conv_shape = (None, None, model_feat_conv.feat_channel)
    model_RPN.rpn(feat_conv_shape, num_anchors)

    _, rpn_cls_score_reshape, rpn_cls_prob, rpn_bbox_pred \
    = model_RPN.model_seq(feat_conv)

    # testing model
    _, rpn_cls_score_reshape_test, rpn_cls_prob_test, rpn_bbox_pred_test \
    = model_RPN.model_seq(feat_conv_test)    
    model_RPN.model_seq.name="RPN"

    # RPN to RegNet for training
    net_build_train = NetBuild('TRAIN', num_classes, cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS)
    rpn_cls_score_shape = (None, None, num_anchors*2)
    rpn_cls_prob_shape = rpn_cls_score_shape
    rpn_bbox_pred_shape = (None, None, num_anchors*4)
    net_build_train.build_roi_ouput_train(feat_conv_shape, im_info_input, gt_box_input, \
                                          rpn_cls_prob_shape, rpn_bbox_pred_shape)
    rpn_labels, rpn_bbox_targets, \
    rpn_bbox_inside_weights, rpn_bbox_outside_weights, \
    rois, labels, bbox_targets, \
    bbox_inside_weights, bbox_outside_weights, \
    pool5 = net_build_train.model_seq([feat_conv, im_info_input, gt_box_input, \
                                       rpn_cls_prob, rpn_bbox_pred])
    net_build_train.model_seq.name="build_RPN_train"
    # RPN to RegNet for testing
    net_build_test = NetBuild('TEST', num_classes, cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS)
    net_build_test.build_roi_ouput_test(feat_conv_shape, im_info_input_test, \
                                        rpn_cls_prob_shape, rpn_bbox_pred_shape)
    rois_test, pool5_test = net_build_test.model_seq([feat_conv_test, im_info_input_test, \
                                                 rpn_cls_prob_test, rpn_bbox_pred_test])
    net_build_test.model_seq.name="build_RPN_test"



    # RegNet
    model_reg = RegNet()
    model_reg.reg_net((cfg.POOLING_SIZE, cfg.POOLING_SIZE, cfg.RPN_CHANNELS), num_classes)

    model_reg.load_weight(pre_net_path)
    cls_score, cls_prob, bbox_pred = model_reg.model_seq(pool5)
    # testing model
    cls_score_test, cls_prob_test, bbox_pred_test = model_reg.model_seq(pool5_test)
    model_reg.model_seq.name="RegNet"



    # losses
    losses = Losses()
    rpn_cls_score_reshape_shape = (None, None, 2)
    rpn_labels_shape = (1, None, None)
    rpn_bbox_targets_shape = rpn_bbox_pred_shape
    rpn_bbox_inside_weights_shape = rpn_bbox_pred_shape
    rpn_bbox_outside_weights_shape = rpn_bbox_pred_shape
    cls_score_shape = (num_classes, )
    labels_shape = (1, )
    bbox_pred_shape = (num_classes*4, )
    bbox_targets_shape = bbox_pred_shape
    bbox_inside_weights_shape = bbox_pred_shape
    bbox_outside_weights_shape = bbox_pred_shape

    losses.build_losses(rpn_cls_score_reshape_shape, rpn_labels_shape, rpn_bbox_pred_shape, \
                        rpn_bbox_targets_shape, rpn_bbox_inside_weights_shape, \
                        rpn_bbox_outside_weights_shape, cls_score_shape, labels_shape, \
                        bbox_pred_shape, bbox_targets_shape, \
                        bbox_inside_weights_shape, bbox_outside_weights_shape)



    rpn_cross_entropy, rpn_loss_box, \
    cross_entropy, loss_box \
    = losses.model_seq([rpn_cls_score_reshape, rpn_labels, rpn_bbox_pred, \
                        rpn_bbox_targets, rpn_bbox_inside_weights, \
                        rpn_bbox_outside_weights, cls_score, labels, \
                        bbox_pred, bbox_targets, bbox_inside_weights, \
                        bbox_outside_weights])


    # build the training model
    model_train = Model(inputs=[image_input, im_info_input, gt_box_input], \
                        outputs=[rpn_cross_entropy, rpn_loss_box, \
                                 cross_entropy, loss_box])

     # build the testing model
    model_test = Model(inputs=[image_input_test, im_info_input_test], \
                       outputs=[rois_test, cls_score_test, cls_prob_test, bbox_pred_test])


    # optimizer and complie the training model

    optimizer = Adam(lr=1e-5)
    def identity_loss(y_true, y_pred):
        return y_pred
    
    model_train.compile(optimizer=optimizer, \
                        loss=[identity_loss, identity_loss, \
                              identity_loss, identity_loss], \
                        loss_weights=[1., 2., 1., 2.])
    # --------------------build the model-------------------


    # load a training check point
    #init_epoch = 1
    #model_train.load_weights('../model_save/model_train_' + str(init_epoch) + '.h5', by_name=True) 
    #optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    return model_train, model_test



def train_model(imdb, roidb, model_train, valroidb, model_vol, output_dir, max_iters=40000):
    # --------------------data gen-------------------
    data_train = DataGen(imdb.num_classes, shuffle=True)
    data_train_gen = data_train.generator(roidb)
    data_val = DataGen(imdb.num_classes, shuffle=False)
    data_val_gen = data_val.generator(valroidb)
    # --------------------data gen-------------------

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #--------------------start training-------------------
    init_epoch = 0
    do_val = True
    fake_label = np.array([0])
    num_epochs = 7
    epoch_length = max_iters
    val_length = len(valroidb)
    for i_epoch in xrange(init_epoch, num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(i_epoch + 1, num_epochs))
        print('lr: ', K.eval(model_train.optimizer.lr))

        iter_num = 0
        loss_epoch = np.zeros([1, 5], dtype=np.float32)
        for i_batch in xrange(epoch_length):

            input_image, im_info, gt_boxes = next(data_train_gen)

            '''
            child_conn_data_gen.send([False])
            lock_sync_data_gen.release()
            input_image, im_info, gt_boxes = child_conn_data_gen.recv()
            '''
            
            loss_train = model_train.train_on_batch([input_image, im_info, gt_boxes], \
                                                    [fake_label, fake_label, fake_label, fake_label])
            
            loss_epoch[0, :] = loss_epoch[0, :] + np.array(loss_train)
            iter_num = iter_num + 1
            progbar.update(iter_num, [('rpn_cls', loss_train[1]), \
                                      ('rpn_reg', loss_train[2]), \
                                      ('detector_cls', loss_train[3]), \
                                      ('detector_reg', loss_train[4]), \
                                      ('loss_sum', loss_train[0])])


        print('training loss:')
        print(('rpn_cls: {}'.format(loss_epoch[0, 1]/iter_num)), \
              ('\nrpn_reg: {}'.format(loss_epoch[0, 2]/iter_num)), \
              ('\ndetector_cls: {}'.format(loss_epoch[0, 3]/iter_num)), \
              ('\ndetector_reg: {}'.format(loss_epoch[0, 4]/iter_num)), \
              ('\nloss_sum: {}'.format(loss_epoch[0, 0]/iter_num)))


        if((i_epoch + 1)%5 == 0):
            model_train.save_weights(output_dir + '/model_train' + str(i_epoch + 1) + datetime.now().strftime("%Y%m%d") + '.h5')
            model_vol.save_weights(output_dir + '/model_test' + str(i_epoch + 1) + datetime.now().strftime("%Y%m%d") + '.h5')

        # change the lr. Seem to be useless
        if((i_epoch + 1)%10 == 0):
            lr_ratio = 0.1
            lr_old = model_train.optimizer.lr
            model_train.optimizer.lr = (lr_old*lr_ratio)

        model_train.save_weights(output_dir + '/model_train.h5')
        model_vol.save_weights(output_dir + '/model_test.h5')
        

        # ----------------------val---------------------------
        if(do_val):
            progbar_val = generic_utils.Progbar(val_length)
            iter_num_val = 0
            loss_val_all = np.zeros([1, 4], dtype=np.float32)
            for i_val in xrange(val_length):
                input_image, im_info, gt_boxes = next(data_val_gen)
                loss_val = model_train.predict_on_batch([input_image, im_info, gt_boxes])

                
                loss_val_all[0, :] = loss_val_all[0, :] + np.array(loss_val)
                iter_num_val = iter_num_val + 1
                progbar_val.update(iter_num_val, [('rpn_cls', loss_val[0]), \
                                                  ('rpn_reg', loss_val[1]), \
                                                  ('detector_cls', loss_val[2]), \
                                                  ('detector_reg', loss_val[3]), \
                                                  ('loss_sum', np.sum(loss_val))])
            print('val_loss:')
            print(('rpn_cls: {}'.format(loss_val_all[0, 0]/iter_num_val)), \
                  ('\nrpn_reg: {}'.format(loss_val_all[0, 1]/iter_num_val)), \
                  ('\ndetector_cls: {}'.format(loss_val_all[0, 2]/iter_num_val)), \
                  ('\ndetector_reg: {}'.format(loss_val_all[0, 3]/iter_num_val)), \
                  ('\nloss_sum: {}'.format(np.sum(loss_val_all[0, :])/iter_num_val)))
        # ----------------------val---------------------------
