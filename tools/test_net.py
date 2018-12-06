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

from nets.VGG16 import VGG16

from datasets.read_db import read_db


# this files are temporary and may need update
from config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir

from model.test_model import test_model, build_model


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import os
import sys
import argparse
import datetime
import pprint

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import pickle


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='model',
            help='model to test',
            default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
 
def read_datasets(args):
    imdb, roidb = read_db(args.imdb_name)
    #print('{:d} roidb entries'.format(len(roidb)))

    # output directory where the models are saved
    #output_dir = get_output_dir(imdb, args.tag)
    #output_dir = args.model
    #print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    #imdb, valroidb = read_db(args.imdbval_name)

    print('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip

    return imdb, roidb

def load_network(args):

    # load network
    if args == 'vgg16':
        net = VGG16()
    #elif args == 'res50':

    #elif args == 'res101':

    #elif args == 'res152':

    else:
        raise NotImplementedError

    return net


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


def load_net_weights(args):

    if(args == 'vgg16'):
        net_file_path = './net_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

    return net_file_path

def load_network(args):

    # load network
    if args == 'vgg16':
        net = VGG16()
    #elif args == 'res50':
    #    net = res50()
    #elif args == 'res101':
    #    net = res101()
    #elif args == 'res152':
    #    net = res101()
    #elif args == 'mobile':
    #    net = mobile()
    else:
        raise NotImplementedError

    return net



if __name__ == '__main__':
    #verbose = True
    verbose = False

    class_name = ('__background__',  # always index 0
                 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)


    # read datasets ---------------------------
    imdb, valroidb = read_datasets(args)

    valroidb = filter_roidb(valroidb)

    # config gpu
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # -------------------build model-----------------
    # load networks ---------------------------
    model_net = load_network(args.net) 
    # pre-training weights file path
    pre_net_file_path = args.model

    model_test = build_model(imdb, model_net, pre_net_file_path)
    # -------------------build model-----------------

    # --------------------testing-------------------
    test_model(imdb, valroidb, model_test, output_dir)


    # --------------------testing-------------------












