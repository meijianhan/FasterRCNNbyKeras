from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


'''Test All'''

import init_path
import keras
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from keras.optimizers import Adam

from network.VGG16 import VGG16
from network.RPN import RPN
from network.RegNet import RegNet
from network.NetBuild import NetBuild
from loss.Losses import Losses
from data_gen.DataGen import DataGen


# this files are temporary and may need update
from config import cfg, cfg_from_file, cfg_from_list
from read_db import read_db
from utils.bbox_transform import clip_boxes, bbox_transform_inv
from utils.nms_wrapper import nms


# set GPU ID
import os
import sys
import argparse
import datetime

def parse_args():
    """
    Parse input arguments
    """
    
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='gpu id: 0, 1, ...',
                        default='0', type=str)
    parser.add_argument('--datasets', dest='datasets',
                        help='voc, voc0712, coco2014',
                        default='voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='vgg16', type=str)
 
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

args = parse_args()
print ("GPU ID: "+args.gpu_id)
# --------------------Set GPU ID ----------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# --------------------End  ----------------------------

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import pickle



def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def select_datasets(String args):
    """
    Extract input arguments
    """
    datasets_args = {'imdb_name': 'voc_2007_trainval',
                    'imdbval_name': 'voc_2007_test',
                    'anchors': '[8,16,32]',
                    'ratios': '[0.5,1,2]',
                    'stepsize': '[50000]',
                    'max_iters': 70000}

    # args #2 Datasets
    if(args ==  'voc'):   
        datasets_args['imdb_name'] = 'voc_2007_trainval'
        datasets_args['imdbval_name'] = 'voc_2007_test'
        datasets_args['anchors'] = '[8,16,32]'
        datasets_args['ratios'] = '[0.5,1,2]'
        datasets_args['stepsize'] = '[50000]'
        datasets_args['max_iters'] = 70000
    elif(args ==  'voc0712'): 
        datasets_args['imdb_name'] = 'voc_2007_trainval+voc_2012_trainval'
        datasets_args['imdbval_name'] = 'voc_2007_test'
        datasets_args['anchors'] = '[8,16,32]'
        datasets_args['ratios'] = '[0.5,1,2]'
        datasets_args['stepsize'] = '[80000]'
        datasets_args['max_iters'] = 110000

    elif(args ==  'coco2014'): 
        datasets_args['imdb_name'] = 'coco_2014_train+coco_2014_valminusminival'
        datasets_args['imdbval_name'] = 'coco_2014_minival'
        datasets_args['anchors'] = '[4,8,16,32]'
        datasets_args['ratios'] = '[0.5,1,2]'
        datasets_args['stepsize'] = '[350000]'
        datasets_args['max_iters'] = 490000

    else:
        print('No dataset given')
        sys.exit(1)

    return datasets_args

def select_net(args):

    if(args == 'vgg16'):
        net = './network/vgg16.yml'
    elif(args == 'res50'):
        net = './network/res50.yml'
    elif(args == 'res101'):
        net = './network/res101.yml'
    elif(args == 'res152'):
        net = './network/res152.yml'
    elif(args == 'mobile'):
        net = './network/mobile.yml'
    else: 
        print('No net given')
        sys.exit(1)
    return net

def load_net_weights(args):

    if(args == 'vgg16'):
        net_address = './network/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    elif(args == 'res50'):
        net_address = './network/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    elif(args == 'res101'):
        net_address = './network/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    elif(args == 'res152'):
        net_address = './network/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    elif(args == 'mobile'):
        net_address = './network/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    else: 
        print ('The net_weights\' file is not exit!')
        sys.exit(1)
    return net_address

def load_network(args):

    # load network
    if args == 'vgg16':
        net = vgg16()
    #elif args == 'res50':
    #    net = vgg16()
    #elif args == 'res101':
    #    net = vgg16()
    #elif args == 'res152':
    #    net = vgg16()
    #elif args == 'mobile':
    #    net = vgg16()
    else:
        raise NotImplementedError

    return net

def main():

    #verbose = True
    verbose = False

    class_name = ('__background__',  # always index 0
                 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')

    # --------------------build the data-------------------
    datasets_args = select_datasets(args.datasets)
    print('Datasets Args: ')
    print(datasets_args)

    cfg_from_file(select_net(args.net))
    cfg_from_list(['ANCHOR_SCALES', datasets_args['anchors'], 'ANCHOR_RATIOS',datasets_args['ratios'], 'TRAIN.STEPSIZE', datasets_args['stepsize']])
    np.random.seed(cfg.RNG_SEED)


    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    
    imdb, valroidb = read_db(datasets_args['imdbval_name'])
    print('{:d} validation roidb entries'.format(len(valroidb)))

    cfg.TRAIN.USE_FLIPPED = orgflip
    imdb.competition_mode('False')
    # ----------------finish building the data---------------

    # --------------------build the model-------------------
    num_classes = imdb.num_classes
    num_anchors = len(cfg.ANCHOR_SCALES)*len(cfg.ANCHOR_RATIOS)
    number_channel = 3

    input_shape = (None, None, number_channel)
    image_input = Input(shape=input_shape)
    image_input_test = Input(shape=input_shape, name='image_input_test')
    im_info_input = Input(shape=(3, ))
    im_info_input_test = Input(shape=(3, ), name='im_info_input_test')
    gt_box_input = Input(shape=(None, 5))


    # convolution feature
    conv_feat_channel = 512
    

    model_feat_conv = load_network(args.net)
    
    model_feat_conv.conv_feature(input_shape, conv_feat_channel)
    # testing model
    feat_conv_test = model_feat_conv.model_seq(image_input_test)
    model_feat_conv.model_seq.name="feat_conv"

    # RPN
    model_RPN = RPN()
    feat_conv_shape = (None, None, model_feat_conv.feat_channel)
    model_RPN.rpn(feat_conv_shape, num_anchors)
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
    # RPN to RegNet for testing
    net_build_test = NetBuild('TEST', num_classes, cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS)
    net_build_test.build_roi_ouput_test(feat_conv_shape, im_info_input_test, \
                                        rpn_cls_prob_shape, rpn_bbox_pred_shape)
    rois_test, pool5_test = net_build_test.model_seq([feat_conv_test, im_info_input_test, \
                                                 rpn_cls_prob_test, rpn_bbox_pred_test])
    net_build_test.model_seq.name="build_RPN_test"

    # RegNet
    model_reg = RegNet()
    model_reg.reg_net((cfg.POOLING_SIZE, cfg.POOLING_SIZE, conv_feat_channel), num_classes)
    # testing model
    cls_score_test, cls_prob_test, bbox_pred_test = model_reg.model_seq(pool5_test)
    model_reg.model_seq.name="RegNet"



    # build the testing model
    model_test = Model(inputs=[image_input_test, im_info_input_test], \
                       outputs=[rois_test, cls_score_test, cls_prob_test, bbox_pred_test])
    model_test.load_weights('./model_save/model_train' + args.datasets + '.h5', by_name=True)
    # --------------------build the model-------------------


    # --------------------data gen-------------------
    data_test = DataGen(num_classes, shuffle=False)
    data_test_gen = data_test.generator(valroidb)
    # --------------------data gen-------------------

    
    # --------------------start testing-------------------
    all_boxes = [[[] for _ in range(len(imdb.image_index))]
                 for _ in range(imdb.num_classes)]
    output_dir = './test_save/'
    epoch_length = len(valroidb)
    thresh = 0.
    max_per_image = 100
    progbar = generic_utils.Progbar(epoch_length)
    for i_batch in xrange(epoch_length):

        input_image, im_info, gt_boxes = next(data_test_gen)
        
        rois_test, _, cls_prob_test, bbox_pred_test \
        = model_test.predict_on_batch([input_image, im_info])

        im_scale = im_info[0, 2]

        stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes))
        means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes))
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
        for j in range(1, imdb.num_classes):
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


        if(verbose):
            print(valroidb[i_batch]['image'])
            image_read = cv2.imread(valroidb[i_batch]['image'])
            for j in range(1, imdb.num_classes):
                bboxes = all_boxes[j][i_batch]
                if(len(bboxes) != 0):
                    for bbox in bboxes:
                        cv2.rectangle(image_read, \
                                      (bbox[0], bbox[1]), \
                                      (bbox[2], bbox[3]), \
                                      (0, 0, 255))
                    print(class_name[j])
                    print(bboxes)

            cv2.imshow('image_read', image_read)
            cv2.waitKey(0)

        progbar.update(i_batch)
    # --------------------start testing-------------------
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    main()













