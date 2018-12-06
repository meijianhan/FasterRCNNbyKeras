from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


'''RPN'''


import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Lambda
from keras.activations import softmax
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Add, Reshape
from keras.layers import Conv2D, MaxPooling2D, Activation, UpSampling2D
from keras.initializers import RandomNormal, Constant

import tensorflow as tf


class RPN(object):
    def __init__(self, is_trainable=True):
        self.is_trainable = is_trainable

    
    def _Lamdba_softmax_layer(self, score_Lambda, name):
        def _softmax_layer(inputs):
            score = inputs
            input_shape = tf.shape(score)
            bottom_reshaped = tf.reshape(score, [-1, input_shape[-1]])
            #bottom_reshaped = tf.reshape(score, [-1, 2])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return Lambda(_softmax_layer)(score_Lambda)
    

    
    def _Lambda_reshape_layer(self, bottom_Lambda, num_dim, name):
        def _reshape_layer(inputs):
            bottom = inputs
            input_shape = tf.shape(bottom)
            with tf.variable_scope(name) as scope:
                # change the channel to the caffe format
                to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
                # then force it to have channel 2
                reshaped = tf.reshape(to_caffe,
                                    tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
                # then swap the channel back
                to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf
        return Lambda(_reshape_layer)(bottom_Lambda)
    

    def _Lambda_argmax_layer(self, score_Lambda, name):
        def _argmax_layer(inputs):
            score = inputs
            return tf.argmax(tf.reshape(score, [-1, 2]), axis=1, name=name)
        return Lambda(_argmax_layer)(score_Lambda)

    def rpn(self, input_shape, num_anchors):

        regu = regularizers.l2(0.0001)

        feat_conv = Input(shape=input_shape)

        feat_anchor = Conv2D(512, (3, 3), kernel_regularizer=regu, padding='same', activation='linear', \
                             kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None), \
                             bias_initializer=Constant(value=0), \
                             name='rpn_conv1')(feat_conv)
        #feat_anchor = BatchNormalization()(feat_anchor)
        feat_anchor = Activation('relu')(feat_anchor)

        rpn_cls_score = Conv2D(num_anchors*2, (1, 1), kernel_regularizer=regu, activation='linear', \
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None), \
                               bias_initializer=Constant(value=0), \
                               name='rpn_out_class')(feat_anchor)
        #rpn_cls_prob = self._Lamdba_softmax_layer(rpn_cls_score, 'rpn_cls_softmax')


        rpn_cls_score_reshape = self._Lambda_reshape_layer(rpn_cls_score, 2, 'rpn_cls_reshape')
        rpn_cls_prob_reshape = self._Lamdba_softmax_layer(rpn_cls_score_reshape, 'rpn_cls_softmax')
        #rpn_cls_pred = self._Lambda_argmax_layer(rpn_cls_score_reshape, 'rpn_cls_argmax')
        rpn_cls_prob = self._Lambda_reshape_layer(rpn_cls_prob_reshape, num_anchors*2, 'rpn_cls_reshape2')
        

        #rpn_bbox_pred = Conv2D(num_anchors*4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(feat_anchor)
        rpn_bbox_pred = Conv2D(num_anchors*4, (1, 1), kernel_regularizer=regu, activation='linear', \
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None), \
                               bias_initializer=Constant(value=0), \
                               name='rpn_out_regress')(feat_anchor)

        
        self.model_seq = Model(inputs=feat_conv, outputs=[rpn_cls_score,
                                                          rpn_cls_score_reshape,
                                                          rpn_cls_prob,
                                                          rpn_bbox_pred])
        '''
        self.model_seq = Model(inputs=feat_conv, outputs=[rpn_cls_score,
                                                          rpn_cls_prob,
                                                          rpn_bbox_pred])
        '''
        #return rpn_cls_score, rpn_bbox_pred
        #return self.model_seq





