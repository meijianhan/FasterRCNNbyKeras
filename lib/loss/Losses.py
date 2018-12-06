from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


'''Losses'''



from keras import backend as K
from keras.models import Sequential, Model
from keras.objectives import categorical_crossentropy
from keras.layers import Lambda, Concatenate, TimeDistributed, Input

import tensorflow as tf


class Losses(object):
    def __init__(self):
        self.sigma_rpn = 3.0

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, \
                        bbox_inside_weights, bbox_outside_weights, \
                        sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
        return loss_box

    def build_RPN_class_loss(self, rpn_cls_score_input, rpn_label_input):
        def _RPN_class_loss(inputs):
            rpn_cls_score, rpn_labels = inputs
            # RPN, class loss
            rpn_labels = tf.round(rpn_labels, name='round_before_cast_RPN')
            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_labels = tf.reshape(rpn_labels, [-1])
            rpn_select = tf.where(tf.not_equal(rpn_labels, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))    
            return rpn_cross_entropy
        return Lambda(_RPN_class_loss)([rpn_cls_score_input, rpn_label_input])


    def build_RPN_bbox_loss(self, rpn_bbox_pred_input, \
                                  rpn_bbox_targets_input, \
                                  rpn_bbox_inside_weights_input, \
                                  rpn_bbox_outside_weights_input):
        def _RPN_bbox_loss(inputs):
            rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = inputs
            # RPN, bbox loss
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=self.sigma_rpn, dim=[1, 2, 3])
            return rpn_loss_box
        return Lambda(_RPN_bbox_loss)([rpn_bbox_pred_input, \
                                      rpn_bbox_targets_input, \
                                      rpn_bbox_inside_weights_input, \
                                      rpn_bbox_outside_weights_input])

    def build_RCNN_class_loss(self, cls_score_input, label_input):
        def _RCNN_class_loss(inputs):
            cls_score, labels = inputs
            # RCNN, class loss
            labels = tf.round(labels, name='round_before_cast_class')
            labels = tf.to_int32(labels, name="labels_to_int32")
            labels = tf.reshape(labels, [-1])
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=labels))
            return cross_entropy
        return Lambda(_RCNN_class_loss)([cls_score_input, label_input])

    def build_RCNN_bbox_loss(self, bbox_pred_input, \
                                   bbox_targets_input, \
                                   bbox_inside_weights_input, \
                                   bbox_outside_weights_input):
        def _RCNN_bbox_loss(inputs):
            bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights = inputs
            # RCNN, bbox loss
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
            return loss_box
        return Lambda(_RCNN_bbox_loss)([bbox_pred_input, \
                                       bbox_targets_input, \
                                       bbox_inside_weights_input, \
                                       bbox_outside_weights_input])


    def build_losses(self, rpn_cls_score_reshape_shape, rpn_labels_shape, rpn_bbox_pred_shape, \
                           rpn_bbox_targets_shape, rpn_bbox_inside_weights_shape, \
                           rpn_bbox_outside_weights_shape, cls_score_shape, labels_shape, \
                           bbox_pred_shape, bbox_targets_shape, \
                           bbox_inside_weights_shape, bbox_outside_weights_shape):

        rpn_cls_score = Input(shape=rpn_cls_score_reshape_shape)
        rpn_labels = Input(shape=rpn_labels_shape)
        rpn_bbox_pred = Input(shape=rpn_bbox_pred_shape)
        rpn_bbox_targets = Input(shape=rpn_bbox_targets_shape)
        rpn_bbox_inside_weights = Input(shape=rpn_bbox_inside_weights_shape)
        rpn_bbox_outside_weights = Input(shape=rpn_bbox_outside_weights_shape)
        cls_score = Input(shape=cls_score_shape)
        labels = Input(shape=labels_shape)
        bbox_pred = Input(shape=bbox_pred_shape)
        bbox_targets = Input(shape=bbox_targets_shape)
        bbox_inside_weights = Input(shape=bbox_inside_weights_shape)
        bbox_outside_weights = Input(shape=bbox_outside_weights_shape)


        rpn_cross_entropy = self.build_RPN_class_loss(rpn_cls_score, rpn_labels)
        rpn_loss_box = self.build_RPN_bbox_loss(rpn_bbox_pred, \
                                              rpn_bbox_targets, \
                                              rpn_bbox_inside_weights, \
                                              rpn_bbox_outside_weights)
        cross_entropy = self.build_RCNN_class_loss(cls_score, labels)
        loss_box = self.build_RCNN_bbox_loss(bbox_pred, \
                                           bbox_targets, \
                                           bbox_inside_weights, \
                                           bbox_outside_weights)

        self.model_seq = Model(inputs=[rpn_cls_score, rpn_labels, rpn_bbox_pred, \
                                       rpn_bbox_targets, rpn_bbox_inside_weights, \
                                       rpn_bbox_outside_weights, cls_score, labels, \
                                       bbox_pred, bbox_targets, bbox_inside_weights, \
                                       bbox_outside_weights], \
                               outputs=[rpn_cross_entropy, rpn_loss_box, \
                                        cross_entropy, loss_box])



