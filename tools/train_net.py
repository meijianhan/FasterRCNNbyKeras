from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


'''Train All'''


import init_path
from nets.VGG16 import VGG16

from model.train_model import train_model, build_model

# this files are temporary and need update
from config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.read_db import read_db


# set GPU ID
import os
import sys
import argparse
import datetime
import pprint

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from multiprocessing import Process, Lock, Pipe
from time import sleep


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
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
    print('{:d} roidb entries'.format(len(roidb)))

    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    #tb_dir = get_output_tb_dir(imdb, args.tag)
    #print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    _, valroidb = read_db(args.imdbval_name)
    print('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip

    return imdb, roidb, valroidb, output_dir

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

def filter_roidb(roidb):
    """
    Remove roidb entries that have no usable RoIs.
    """

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
    return filtered_roidb

def load_net_weights(args):

    if(args == 'vgg16'):
        # check
        # Download
        # load path
        net_file_path = './net_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    #elif(args == 'res50'):


    #elif(args == 'res101'):

    else: 
        print ('The net_weights\' file is not exit!')
        sys.exit(1)
    return net_file_path


if __name__ == '__main__':
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
    imdb, roidb, valroidb, output_dir = read_datasets(args)
    

    # Remove roidb entries that have no usable RoIs.
    roidb = filter_roidb(roidb)
    valroidb = filter_roidb(valroidb)

    # config gpu
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True


    # -------------------build model-----------------
    # load networks ---------------------------
    model_net = load_network(args.net) 
    # pre-training weights
    pre_net_file_path = load_net_weights(args.net)

    model_train, model_vol = build_model(imdb, model_net, pre_net_file_path)
    # -------------------build model-----------------


    # --------------------training-------------------
    train_model(imdb, roidb, model_train, valroidb, model_vol, output_dir, max_iters=args.max_iters)

    # --------------------training-------------------













