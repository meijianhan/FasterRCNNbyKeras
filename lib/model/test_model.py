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
from os import listdir
from os.path import isfile, join
from multiprocessing import Process, Lock, Pipe
from time import sleep
import pickle
import os


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

    #test sets
    image_input_test = Input(shape=input_shape, name='image_input_test')
    im_info_input_test = Input(shape=(3, ), name='im_info_input_test')

    #--------- Conv Layer, output = feature maps
    # convolution feature
    model_feat_conv = net    
    model_feat_conv.conv_feature(input_shape, cfg.RPN_CHANNELS)
    model_feat_conv.load_weight(pre_net_path)

    #feat_conv = model_feat_conv.model_seq(image_input)
    # testing model
    feat_conv_test = model_feat_conv.model_seq(image_input_test)

    model_feat_conv.model_seq.name="feat_conv"

    #-------- Region Proposal Network (RPN), output = proposals
    # RPN
    model_RPN = RPN()
    feat_conv_shape = (None, None, model_feat_conv.feat_channel)
    model_RPN.rpn(feat_conv_shape, num_anchors)

    #_, rpn_cls_score_reshape, rpn_cls_prob, rpn_bbox_pred \
    #= model_RPN.model_seq(feat_conv)

    # testing model
    _, rpn_cls_score_reshape_test, rpn_cls_prob_test, rpn_bbox_pred_test \
    = model_RPN.model_seq(feat_conv_test)    
    model_RPN.model_seq.name="RPN"


    rpn_cls_score_shape = (None, None, num_anchors*2)
    rpn_cls_prob_shape = rpn_cls_score_shape
    rpn_bbox_pred_shape = (None, None, num_anchors*4)

    
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
    
    #cls_score, cls_prob, bbox_pred = model_reg.model_seq(pool5)
    # testing model
    cls_score_test, cls_prob_test, bbox_pred_test = model_reg.model_seq(pool5_test)
    model_reg.model_seq.name="RegNet"



     # build the testing model
    model_test = Model(inputs=[image_input_test, im_info_input_test], \
                       outputs=[rois_test, cls_score_test, cls_prob_test, bbox_pred_test])

    model_test.load_weights(pre_net_path, by_name=True)

    return model_test


def test_model(imdb, valroidb, model_test, output_dir):
    # --------------------data gen-------------------
    data_test = DataGen(imdb.num_classes, shuffle=False)
    data_test_gen = data_test.generator(valroidb)
    # --------------------data gen-------------------

    #verbose = False

    #class_name = ('__background__',  # always index 0
    #             'aeroplane', 'bicycle', 'bird', 'boat',
    #             'bottle', 'bus', 'car', 'cat', 'chair',
    #             'cow', 'diningtable', 'dog', 'horse',
    #             'motorbike', 'person', 'pottedplant',
    #             'sheep', 'sofa', 'train', 'tvmonitor')


    # --------------------start testing-------------------
    all_boxes = [[[] for _ in range(len(imdb.image_index))]
                 for _ in range(imdb.num_classes)]
    #output_dir = '../test_save/'
    epoch_length = len(imdb.image_index)
    thresh = 0.
    max_per_image = 100
    progbar = generic_utils.Progbar(epoch_length)
    for i_batch in xrange(epoch_length):

        input_image, im_info, gt_boxes = next(data_test_gen)
        
        rois_test, _, cls_prob_test, bbox_pred_test \
        = model_test.predict_on_batch([input_image, im_info])

        im_scale = im_info[0, 2]

        stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (imdb.num_classes))
        means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (imdb.num_classes))
        bbox_pred_test *= stds
        bbox_pred_test += means

        boxes = rois_test[:, 1:5]/im_scale
        scores = np.reshape(cls_prob_test, [cls_prob_test.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred_test, [bbox_pred_test.shape[0], -1])
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            image_shape = np.floor(im_info[0, 0:2]/im_scale)
            pred_boxes = _clip_boxes(pred_boxes, image_shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))


        # skip j = 0, because it's the background class
        for j in range(1, imdb.image_index):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                         .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i_batch] = cls_dets


        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i_batch][:, -1]
                                     for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i_batch][:, -1] >= image_thresh)[0]
                    all_boxes[j][i_batch] = all_boxes[j][i_batch][keep, :]

        progbar.update(i_batch)
    # --------------------start testing-------------------

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
