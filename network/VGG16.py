from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


'''VGG convolution feature layers'''


import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Lambda
from keras.activations import softmax
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Add, Reshape
from keras.layers import Conv2D, MaxPooling2D, Activation, UpSampling2D


class VGG16(object):
    def __init__(self, is_trainable=True):
        self.is_trainable = is_trainable

    def load_weight(self, weight_path):
        self.model_seq.load_weights(weight_path, by_name=True)

    def conv_feature(self, input_shape, output_channel):

        image_input = Input(shape=input_shape)

        regu = regularizers.l2(0.0001)

        # With BN
        feat_conv = Conv2D(64, (3,3), kernel_regularizer=regu, padding='same', name='block1_conv1', trainable=self.is_trainable)(image_input)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = Conv2D(64, (3,3), kernel_regularizer=regu, padding='same', name='block1_conv2', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = MaxPooling2D((2,2), strides=(2,2), padding='same', name='block1_pool')(feat_conv)

        feat_conv = Conv2D(128, (3,3), kernel_regularizer=regu, padding='same', name='block2_conv1', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = Conv2D(128, (3,3), kernel_regularizer=regu, padding='same', name='block2_conv2', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = MaxPooling2D((2,2), strides=(2,2), padding='same', name='block2_pool')(feat_conv)

        feat_conv = Conv2D(256, (3,3), kernel_regularizer=regu, padding='same', name='block3_conv1', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = Conv2D(256, (3,3), kernel_regularizer=regu, padding='same', name='block3_conv2', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = Conv2D(256, (3,3), kernel_regularizer=regu, padding='same', name='block3_conv3', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = MaxPooling2D((2,2), strides=(2,2), padding='same', name='block3_pool')(feat_conv)

        feat_conv = Conv2D(512, (3,3), kernel_regularizer=regu, padding='same', name='block4_conv1', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = Conv2D(512, (3,3), kernel_regularizer=regu, padding='same', name='block4_conv2', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = Conv2D(512, (3,3), kernel_regularizer=regu, padding='same', activation='linear', name='block4_conv3', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv4 = Activation('relu')(feat_conv)
        feat_conv_pool4 = MaxPooling2D((2,2), strides=(2,2), padding='same', name='block4_pool')(feat_conv4)

        feat_conv = Conv2D(512, (3,3), kernel_regularizer=regu, padding='same', name='block5_conv1', trainable=self.is_trainable)(feat_conv_pool4)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = Conv2D(512, (3,3), kernel_regularizer=regu, padding='same', name='block5_conv2', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv = Activation('relu')(feat_conv)
        feat_conv = Conv2D(output_channel, (3,3), kernel_regularizer=regu, padding='same', activation='linear', name='block5_conv3', trainable=self.is_trainable)(feat_conv)
        #feat_conv = BatchNormalization()(feat_conv)
        feat_conv5 = Activation('relu')(feat_conv)


        #self.model_seq = Model(inputs=image_input, outputs=[feat_conv4, feat_conv5])
        self.model_seq = Model(inputs=image_input, outputs=feat_conv5)
        self.feat_channel = output_channel

        #return self.model_seq



