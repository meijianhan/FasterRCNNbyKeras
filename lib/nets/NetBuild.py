from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


'''TODO'''


import keras
from keras.models import Sequential, Model
from keras.layers import Lambda, TimeDistributed, Input

import tensorflow as tf
import tensorflow.contrib.slim as slim

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from config import cfg

import numpy as np


class NetBuild(object):
    def __init__(self, mode, num_classes, \
                anchor_scales, anchor_ratios):
        self._feat_stride = [16, ]
        self._mode = mode
        self._num_classes = num_classes
        self._tag = 'default'
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        self._num_anchors = self._num_scales * self._num_ratios

    def build_proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, anchors, im_info_input, name):

        def compute_output_shape(input_shape):
            return [(cfg.TEST.RPN_TOP_N, 5), (cfg.TEST.RPN_TOP_N, 1)]

        def compute_mask(inputs, mask=None):
            return 2*[None]

        def _proposal_top_layer(inputs):
            rpn_cls_prob, rpn_bbox_pred, build_anchors, im_info = inputs
            im_info = im_info[0]
            with tf.variable_scope(name) as scope:
                if cfg.USE_E2E_TF:
                    rois, rpn_scores = proposal_top_layer_tf(rpn_cls_prob,
                                                          rpn_bbox_pred,
                                                          im_info,
                                                          self._feat_stride,
                                                          build_anchors,
                                                          self._num_anchors)
                else:
                    rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, im_info,
                                           self._feat_stride, build_anchors, self._num_anchors],
                                          [tf.float32, tf.float32], name="proposal_top")
            
                rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
                rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

                return [rois, rpn_scores]
        return Lambda(_proposal_top_layer, \
                      output_shape=compute_output_shape, \
                      mask=compute_mask)([rpn_cls_prob, rpn_bbox_pred, \
                                          anchors, im_info_input])

    def build_proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, anchors, im_info_input, name):

        def compute_output_shape(input_shape):
            return [(None, 5), (None, 1)]

        def compute_mask(inputs, mask=None):
            return 2*[None]

        def _proposal_layer(inputs):
            build_rpn_cls_prob, build_rpn_bbox_pred, build_anchors, im_info = inputs
            im_info = im_info[0]
            with tf.variable_scope(name) as scope:
                if cfg.USE_E2E_TF:
                    rois, rpn_scores = proposal_layer_tf(build_rpn_cls_prob,
                                                      build_rpn_bbox_pred,
                                                      im_info,
                                                      self._mode,
                                                      self._feat_stride,
                                                      build_anchors,
                                                      self._num_anchors)
                else:
                    rois, rpn_scores = tf.py_func(proposal_layer,
                                          [build_rpn_cls_prob, build_rpn_bbox_pred, im_info, self._mode,
                                           self._feat_stride, build_anchors, self._num_anchors],
                                          [tf.float32, tf.float32], name="proposal")

                rois.set_shape([None, 5])
                rpn_scores.set_shape([None, 1])

                return [rois, rpn_scores]
        return Lambda(_proposal_layer, \
                      output_shape=compute_output_shape, \
                      mask=compute_mask)([rpn_cls_prob, rpn_bbox_pred, \
                                          anchors, im_info_input])

    # Only use it if you have roi_pooling op written in tf.image
    def build_roi_pool_layer(self, bottom, rois, name):
        def _roi_pool_layer(inputs):
            bottom, rois = inputs
            with tf.variable_scope(name) as scope:
                return tf.image.roi_pooling(bottom, rois,
                                          pooled_height=cfg.POOLING_SIZE,
                                          pooled_width=cfg.POOLING_SIZE,
                                          spatial_scale=1. / 16.)[0]
        return Lambda(_roi_pool_layer)([bottom, rois])

    def build_crop_pool_layer(self, bottom, rois, name):
        def _crop_pool_layer(inputs):
            bottom, rois = inputs
            with tf.variable_scope(name) as scope:
                batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
                # Get the normalized coordinates of bounding boxes
                bottom_shape = tf.shape(bottom)
                height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
                width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
                x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
                y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
                x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
                y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
                # Won't be back-propagated to rois anyway, but to save time
                bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

                return slim.max_pool2d(crops, [2, 2], padding='SAME')
        return Lambda(_crop_pool_layer)([bottom, rois])

    def build_anchor_target_layer(self, rpn_cls_score, anchors, \
                                  im_info_input, gt_box_input, name):

        def compute_output_shape(input_shape):
            return [(1, 1, None, None), \
                    (1, None, None, self._num_anchors*4), \
                    (1, None, None, self._num_anchors*4), \
                    (1, None, None, self._num_anchors*4)]

        def compute_mask(inputs, mask=None):
            return 4*[None]

        def _anchor_target_layer(inputs):
            rpn_cls_score, build_anchors, im_info, gt_boxes = inputs
            im_info = im_info[0]
            gt_boxes = gt_boxes[0]
            with tf.variable_scope(name) as scope:
                rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                    anchor_target_layer,
                    [rpn_cls_score, gt_boxes, im_info, self._feat_stride, build_anchors, self._num_anchors],
                    [tf.float32, tf.float32, tf.float32, tf.float32],
                    name="anchor_target")

                rpn_labels.set_shape([1, 1, None, None])
                #rpn_labels.set_shape([1, None, None, self._num_anchors])
                rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
                rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
                rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

                return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]
        return Lambda(_anchor_target_layer, \
                      output_shape=compute_output_shape, \
                      mask=compute_mask)([rpn_cls_score, anchors, \
                                            im_info_input, gt_box_input])

    def build_proposal_target_layer(self, rois_input, roi_scores_input, gt_box_input, name):

        def compute_output_shape(input_shape):
            return [(cfg.TRAIN.BATCH_SIZE, 5), \
                    (cfg.TRAIN.BATCH_SIZE, 1), \
                    (cfg.TRAIN.BATCH_SIZE, 1), \
                    (cfg.TRAIN.BATCH_SIZE, self._num_classes*4), \
                    (cfg.TRAIN.BATCH_SIZE, self._num_classes*4), \
                    (cfg.TRAIN.BATCH_SIZE, self._num_classes*4)]

        def compute_mask(inputs, mask=None):
            return 6*[None]

        def _proposal_target_layer(inputs):
            rois_build, roi_scores_build, gt_boxes = inputs
            gt_boxes = gt_boxes[0]
            with tf.variable_scope(name) as scope:
                rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                    proposal_target_layer,
                    [rois_build, roi_scores_build, gt_boxes, self._num_classes],
                    [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                    name="proposal_target")

                rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
                roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
                labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
                bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
                bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
                bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])


                return [rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
        return Lambda(_proposal_target_layer, \
                      output_shape=compute_output_shape, \
                      mask=compute_mask)([rois_input, roi_scores_input, gt_box_input])

    def build_anchor_component(self, im_info_input):
        def _anchor_component(inputs):
            im_info = inputs
            im_info = im_info[0]
            with tf.variable_scope('ANCHOR_' + self._tag) as scope:
                # just to get the shape right
                height = tf.to_int32(tf.ceil(im_info[0] / np.float32(self._feat_stride[0])))
                width = tf.to_int32(tf.ceil(im_info[1] / np.float32(self._feat_stride[0])))

                if cfg.USE_E2E_TF:
                    anchors, anchor_length = generate_anchors_pre_tf(height,
                                                                    width,
                                                                    self._feat_stride,
                                                                    self._anchor_scales,
                                                                    self._anchor_ratios)
                else:
                    anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                        [height, width,
                                                         self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                        [tf.float32, tf.int32], name="generate_anchors")
                anchors.set_shape([None, 4])
                anchor_length.set_shape([])

                return anchors
        return Lambda(_anchor_component)(im_info_input)


    def build_roi_ouput_train(self, feat_conv_shape, im_info_input, gt_box_input, \
                                    rpn_cls_prob_shape, rpn_bbox_pred_shape):
        
        feat_conv = Input(shape=feat_conv_shape)
        #rpn_cls_score = Input(shape=rpn_cls_score_shape)
        rpn_cls_prob = Input(shape=rpn_cls_prob_shape)
        rpn_bbox_pred = Input(shape=rpn_bbox_pred_shape)

        anchors = self.build_anchor_component(im_info_input)
        rois_temp, roi_scores = self.build_proposal_layer(rpn_cls_prob, rpn_bbox_pred, \
                                                            anchors, im_info_input, 'rois')
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights \
        = self.build_anchor_target_layer(rpn_cls_prob, anchors, \
                                         im_info_input, gt_box_input, 'anchor')
        rois, _, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = self.build_proposal_target_layer(rois_temp, roi_scores, \
                                           gt_box_input, 'rpn_rois')
        pool5 = self.build_crop_pool_layer(feat_conv, rois, 'pool5')


        self.model_seq = Model(inputs=[feat_conv, im_info_input, gt_box_input, \
                                       rpn_cls_prob, rpn_bbox_pred], \
                               outputs=[rpn_labels, rpn_bbox_targets, \
                                        rpn_bbox_inside_weights, rpn_bbox_outside_weights, \
                                        rois, labels, bbox_targets, \
                                        bbox_inside_weights, bbox_outside_weights, \
                                        pool5])

    def build_roi_ouput_test(self, feat_conv_shape, im_info_input, \
                                   rpn_cls_prob_shape, rpn_bbox_pred_shape):
        
        feat_conv = Input(shape=feat_conv_shape)
        rpn_cls_prob = Input(shape=rpn_cls_prob_shape)
        rpn_bbox_pred = Input(shape=rpn_bbox_pred_shape)

        anchors = self.build_anchor_component(im_info_input)
        rois, _ = self.build_proposal_layer(rpn_cls_prob, rpn_bbox_pred, \
                                            anchors, im_info_input, 'rois')
        pool5 = self.build_crop_pool_layer(feat_conv, rois, 'pool5')

        self.model_seq = Model(inputs=[feat_conv, im_info_input, \
                                       rpn_cls_prob, rpn_bbox_pred], \
                               outputs=[rois, pool5])

