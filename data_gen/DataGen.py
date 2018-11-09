#!/usr/bin/env python2
# -*- coding: utf-8 -*-



'''Data generator'''

import keras
from keras.preprocessing.image import *
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import numpy as np
import cv2
import scipy.io

from config import cfg

class DataGen(object):
    'Generates data for Keras'
    def __init__(self, number_classes, shuffle):
        'Initialization'
        self.number_classes = number_classes
        self.shuffle = shuffle
        self.batch_size = 1
        self.image_channel = 3


    def generator(self, data_base):
        'Generates batches of samples'
        
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(data_base)
            len_indexes = len(indexes)

            # Generate batches
            Max_index = int(len_indexes/self.batch_size)
            for i_index in range(Max_index):
                # Find list of IDs
                data_base_batch_list = [data_base[j_batch] for j_batch in indexes[i_index*self.batch_size:(i_index + 1)*self.batch_size]]
                # Generate data
                input_image, im_info, gt_boxes = self.__data_generating(data_base_batch_list)
                yield input_image, im_info, gt_boxes



    def __get_exploration_order(self, data_base):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(data_base))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_processing(self):
        pass

    def __data_generating(self, data_base_batch_list):
        'Generates data of BatchSize samples'

        target_size = cfg.TRAIN.SCALES[0]
        max_size = cfg.TRAIN.MAX_SIZE
        for i_count, i_db in enumerate(data_base_batch_list):
            # gt image
            image_read = cv2.imread(i_db['image'])
            #print(i_db['image'])
            # change the color channel to RGB
            image_read = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB)
            if(i_db['flipped']):
                image_read = image_read[:, ::-1, :]
            image_read = image_read.astype(np.float32, copy=False)
            image_read = image_read - cfg.PIXEL_MEANS
            im_shape = image_read.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            im_scale = float(target_size)/float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size)/float(im_size_max)
            image_read = cv2.resize(image_read, None, None, \
                                    fx=im_scale, fy=im_scale, \
                                    interpolation=cv2.INTER_LINEAR)

            # gt boxes
            gt_inds = np.where(i_db['gt_classes'] != 0)[0]
            gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            gt_boxes[:, 0:4] = i_db['boxes'][gt_inds, :]*im_scale
            gt_boxes[:, 4] = i_db['gt_classes'][gt_inds]

        # output format. (only single batch in this version)
        input_image_out = np.zeros((self.batch_size, image_read.shape[0], image_read.shape[1], \
                                    self.image_channel), dtype=np.float32)
        input_image_out[0, :] = image_read
        im_info_out = np.zeros((self.batch_size, 3), dtype=np.float32)
        im_info_out[0, 0] = image_read.shape[0]
        im_info_out[0, 1] = image_read.shape[1]
        im_info_out[0, 2] = im_scale
        gt_boxes_out = np.zeros((self.batch_size, len(gt_inds), 5), dtype=np.float32)
        gt_boxes_out[0, :] = gt_boxes

        return input_image_out, im_info_out, gt_boxes_out








