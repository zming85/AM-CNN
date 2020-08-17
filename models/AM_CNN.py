"""MobileNet v1 models for Keras.
This is a revised implementation from Somshubra Majumdar's SENet repo:
(https://github.com/titu1994/keras-squeeze-excite-network)
# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense,MaxPooling2D
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras.applications import imagenet_utils
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.regularizers import l2
from models.attention_module import attach_attention_module

def AM_CNN(input_shape=None,classes=None,attention_module=None):
    # Determine proper input shape
    # img_input = Input(shape=input_shape)
    #
    # x = Conv2D(16, (5, 5), kernel_initializer='random_normal',name='conv_pw_1')(img_input)
    # # attention_module
    # x = BatchNormalization()(x)
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # x = Conv2D(32, (5, 5), kernel_initializer='random_normal',name='conv_pw_2')(x)
    # # attention_module
    # x = BatchNormalization()(x)
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    #
    # x = Conv2D(128, (6, 6), kernel_initializer='random_normal',name='conv_pw_6')(x)
    # x = BatchNormalization()(x)
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    #
    # x = Conv2D(64, (5, 5),kernel_initializer='random_normal', name='conv_pw_7')(x)
    # x = BatchNormalization()(x)
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(classes, (3, 3), kernel_initializer='random_normal',activation='softmax', name='conv_pw_8')(x)
    # x = Flatten(name='flatten')(x)
    #
    #
    # # Create model.
    # model = Model(img_input, x)
    # return model

    img_input = Input(shape=input_shape)


    # x = Conv2D(16, (5, 5),name='conv_pw_1')(img_input)  # 88
    # # attention_module
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    #
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # x = Conv2D(32, (5, 5),name='conv_pw_2')(x)
    # # attention_module
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    #
    # x = Conv2D(128, (6, 6),name='conv_pw_3')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    #
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    #
    # x = Conv2D(64, (5, 5), name='conv_pw_4')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    #
    # x = Conv2D(classes, (3, 3),activation='softmax', name='conv_pw_5')(x)
    # x = Flatten(name='flatten')(x)
    #
    #
    # # Create model.
    # model = Model(img_input, x)
    # return model



    x = Conv2D(16, (3, 3), kernel_initializer='random_uniform',name='conv_pw_1')(img_input)# 100
    # attention_module
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if attention_module is not None:
        x = attach_attention_module(x, attention_module)


    x = Conv2D(32, (5, 5), kernel_initializer='random_uniform',name='conv_pw_2')(x)
    # attention_module
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if attention_module is not None:
        x = attach_attention_module(x, attention_module)

    x = Conv2D(64, (7, 7), kernel_initializer='random_uniform', name='conv_pw_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module)

    x = Conv2D(128, (5, 5), kernel_initializer='random_uniform', name='conv_pw_4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (5, 5), kernel_initializer='random_uniform',name='conv_pw_5')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (6, 6), kernel_initializer='random_uniform',name='conv_pw_6')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (5, 5), kernel_initializer='random_uniform', name='conv_pw_7')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module)

    x = Conv2D(classes, (3, 3), kernel_initializer='random_uniform',activation='softmax', name='conv_pw_8')(x)
    x = Flatten(name='flatten')(x)


    # Create model.
    model = Model(img_input, x)
    return model





    # x = Conv2D(16, (3, 3),name='conv_pw_1')(img_input)# 100
    # # attention_module
    # x = BatchNormalization()(x)
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(32, (5, 5),name='conv_pw_2')(x)
    # # attention_module
    # x = BatchNormalization()(x)
    #
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(64, (7, 7), name='conv_pw_3')(x)
    # x = BatchNormalization()(x)
    #
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(128, (5, 5), name='conv_pw_4')(x)
    # x = BatchNormalization()(x)
    #
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    #
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # x = Conv2D(256, (5, 5),name='conv_pw_5')(x)
    # x = BatchNormalization()(x)
    #
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    #
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # x = Conv2D(128, (6, 6),name='conv_pw_6')(x)
    # x = BatchNormalization()(x)
    #
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    #
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(64, (5, 5), name='conv_pw_7')(x)
    # x = BatchNormalization()(x)
    #
    # # attention_module
    # if attention_module is not None:
    #     x = attach_attention_module(x, attention_module)
    # x = Activation('relu')(x)
    #
    # x = Conv2D(classes, (3, 3),activation='softmax', name='conv_pw_8')(x)
    # x = Flatten(name='flatten')(x)
    #
    #
    # # Create model.
    # model = Model(img_input, x)
    # return model





# model.add(12,input_dim=8,kernel_initializer='random_uniform')
# 每个神经元可以用特定的权重进行初始化 。 Keras 提供了 几个选择 ， 其中最常用的选择如下所示。
#
# random_unifrom:权重被初始化为（-0.5，0.5）之间的均匀随机的微小数值，换句话说，给定区间里的任何值都可能作为权重 。
# random_normal:根据高斯分布初始化权重，其中均值为0，标准差为0.05。
# zero:所有权重被初始化为0。



