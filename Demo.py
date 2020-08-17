# -*- coding: utf-8 -*-
"""
Created on  Jun  2 21:39:53 2020

@author: Ming Zhang
"""
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from models import AM_CNN
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import confusion_matrix

K.set_image_dim_ordering('tf')
batch_size =16
epochs = 300
num_classes =10
subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy
# Choose what attention_module to use: cbam_block / se_block / None
# attention_module =None
attention_module ='cbam_block'
# attention_module ='se_block'
# Input image dimensions (100, 100, 3)
img_width, img_height = 100,100
# img_width, img_height = 88,88
input_shape = ((img_width, img_height,3))

print('Model loaded.')
model = AM_CNN.AM_CNN(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = AConvNets.AConvNets(input_shape=input_shape, classes=num_classes)

learning_rate=1e-2

model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr=learning_rate,decay=learning_rate/epochs,momentum=0.9),
              metrics=['accuracy'])

print(model.summary())

train_data_dir = 'data/100/SOC/train'
validation_data_dir = 'data/100/SOC/test'
nb_train_samples = 2746
nb_validation_samples = 2426


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,    # 随机错切换角度
        rotation_range=10., # 角度值，0~180，图像旋转
        zoom_range=0.2,     # 随机缩放范围
        horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
# 图片generator
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        figsize = 11, 6
        figure, ax = plt.subplots(figsize=figsize)
        iters = range(len(self.losses[loss_type]))
        # plt.figure()
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 30,
                 }
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 23,
                 }
        # accuracy
        plt.subplot(2, 2, 1)
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')

        plt.tick_params(labelsize=23)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Verdana') for label in labels]
        plt.grid(True)
        plt.title('Model accuracy',font1)  # 设置图表标题
        plt.xlabel(loss_type,font2)
        plt.ylabel('accuracy',font2)
        plt.legend(loc="best", prop=font2)
        plt.tick_params(font1)

        plt.subplot(2, 2, 2)
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        plt.plot(iters, self.val_loss[loss_type], 'b', label='val loss')
        plt.tick_params(labelsize=23)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.grid(True)
        plt.title('Model loss',font1)
        plt.xlabel(loss_type,font2)
        plt.ylabel('loss',font2)
        plt.legend(loc="best", prop=font2)
        plt.savefig('figure.png')
        plt.show()

history = LossHistory()

early_stopping = EarlyStopping(monitor='val_acc', patience=25, verbose=2, mode='max')

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[history])

model.evaluate_generator(validation_generator,verbose=1)

prediction=model.predict_generator(validation_generator,verbose=1)
predict_label=np.argmax(prediction,axis=1)
true_label=validation_generator.classes

confusion_matrix.plot_confusion_matrix(true_label, predict_label, save_flg = True)


history.loss_plot('epoch')
model.save('my_model.h5')
