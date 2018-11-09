from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


'''RegNet'''


import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Lambda
from keras.activations import softmax
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Add, Reshape
from keras.layers import Conv2D, MaxPooling2D, Activation, UpSampling2D
from keras.initializers import RandomNormal, Constant

import tensorflow as tf

class RegNet(object):
    def __init__(self, is_trainable=True):
        self.is_trainable = is_trainable

    def load_weight(self, weight_path):
        self.model_seq.load_weights(weight_path, by_name=True)

    def _Lamdba_softmax_layer(self, score_Lambda, name):
        def _softmax_layer(inputs):
            score = inputs
            return tf.nn.softmax(score, name=name)
        return Lambda(_softmax_layer)(score_Lambda)

    def reg_net(self, input_shape, num_classes):

        regu = regularizers.l2(0.0001)

        roi_pool = Input(input_shape)

        feat_reg = Flatten(name='flatten')(roi_pool)
        feat_reg = Dense(4096, kernel_regularizer=regu, name='fc1')(feat_reg)
        feat_reg = BatchNormalization(name='BN_reg1')(feat_reg)
        feat_reg = Activation('relu')(feat_reg)
        feat_reg = Dropout(0.5)(feat_reg)
        feat_reg = Dense(4096, kernel_regularizer=regu, name='fc2')(feat_reg)
        feat_reg = BatchNormalization(name='BN_reg2')(feat_reg)
        feat_reg = Activation('relu')(feat_reg)
        feat_reg = Dropout(0.5)(feat_reg)

        #cls_score = Dense(num_classes, activation='linear', kernel_initializer='zero')(feat_reg)
        cls_score = Dense(num_classes, kernel_regularizer=regu, activation='linear', \
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None), \
                          bias_initializer=Constant(value=0), \
                          name='cls_score')(feat_reg)
        cls_prob = self._Lamdba_softmax_layer(cls_score, 'cls_score_softmax')
        # note: no regression target for bg class
        #bbox_pred = Dense(4*num_classes, activation='linear', kernel_initializer='zero')(feat_reg)
        bbox_pred = Dense(4*num_classes, kernel_regularizer=regu, activation='linear', \
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=None), \
                          bias_initializer=Constant(value=0), \
                          name='bbox_pred')(feat_reg)

        self.model_seq = Model(inputs=roi_pool, outputs=[cls_score, cls_prob, bbox_pred])

        #return cls_score, bbox_pred
        #return self.model_seq
