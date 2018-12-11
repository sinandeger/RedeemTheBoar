import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from itertools import chain

import skimage.io
import skimage.segmentation
from skimage.transform import resize
from skimage.morphology import label

from scipy.misc import imread
from PIL import Image

import keras
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def mean_iou(y_pred, y_true):
    y_pred_ = tf.to_int64(y_pred > 0.5)
    y_true_ = tf.to_int64(y_true > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

"""Data Augmentation Routines"""


# def train_data_aug(X_data, Y_data, sd, batchsize):
#
#     """Many more arguments can/will be added to the dictionary below"""
#
#     data_gen_args = dict(rotation_range=30.0,
#                          horizontal_flip=True)
#
#     image_augdata = ImageDataGenerator(**data_gen_args)
#     mask_augdata = ImageDataGenerator(**data_gen_args)
#
#     image_augdata.fit(X_data, augment=True, seed=sd)
#     mask_augdata.fit(Y_data, augment=True, seed=sd)
#
#     """Use save_to_dir='/home/sinandeger/PycharmProjects/DataScienceBowl18/Aug_img' to output generated images"""
#
#     image_generator = image_augdata.flow(X_data, batch_size=batchsize, seed=sd, shuffle=True)
#     mask_generator = mask_augdata.flow(Y_data, batch_size=batchsize, seed=sd, shuffle=True)
#
#     aug_generator = zip(image_generator, mask_generator)
#
#     return aug_generator

#
# def train_data_aug(X_data, Y_data, sd, batchsize):
#
#     """Many more arguments can/will be added to the dictionary below"""
#
#     data_gen_args = dict(rotation_range=30.0,
#                          horizontal_flip=True)
#
#     image_augdata = ImageDataGenerator(**data_gen_args)
#     mask_augdata = ImageDataGenerator(**data_gen_args)
#
#     image_augdata.fit(X_data, augment=True, seed=sd)
#     mask_augdata.fit(Y_data, augment=True, seed=sd)
#
#     """Use save_to_dir='/home/sinandeger/PycharmProjects/DataScienceBowl18/Aug_img' to output generated images"""
#
#     image_generator = image_augdata.flow(X_data, batch_size=batchsize, seed=sd, shuffle=True)
#     mask_generator = mask_augdata.flow(Y_data, batch_size=batchsize, seed=sd, shuffle=True)
#
#     aug_generator = zip(image_generator, mask_generator)
#
#     return aug_generator