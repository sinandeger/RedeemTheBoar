import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation
import pandas as pd
import random
from skimage import transform
import os
from tqdm import tqdm
from subprocess import check_output

import tensorflow as tf

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import time

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

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from IPython.display import clear_output

import keras_model_functions as kerfunc

# def read_image_labels(image_id):
#     # most of the content in this function is taken from 'Example Metric Implementation' kernel
#     # by 'William Cukierski'
#     image_file = "/home/sinandeger/kaggle_Competitions/DataScienceBowl18/Train/{}/images/{}.png".format(image_id, image_id)
#     mask_file = "/home/sinandeger/kaggle_Competitions/DataScienceBowl18/Train/{}/masks/*.png".format(image_id)
#     image = skimage.io.imread(image_file)
#     masks = skimage.io.imread_collection(mask_file).concatenate()
#     height, width, _ = image.shape
#     num_masks = masks.shape[0]
#     labels = np.zeros((height, width), np.uint16)
#     for index in range(0, num_masks):
#         labels[masks[index] > 0] = index + 1
#     return image, labels
#
# image_ids = check_output(["ls", "/home/sinandeger/kaggle_Competitions/DataScienceBowl18/Train"]).decode("utf8").split()
# image_id = image_ids[random.randint(0,len(image_ids))]
# image, labels = read_image_labels(image_id)
# plt.subplot(221)
# plt.imshow(image)
# plt.subplot(222)
# plt.imshow(labels)
# plt.show()

# image_width = 128
# image_height = 128
# image_channels = 3
# train_folder = '/home/sinandeger/kaggle_Competitions/DataScienceBowl18/Train/'
# test_folder = '/home/sinandeger/kaggle_Competitions/DataScienceBowl18/Test/'
#
# train_ids = next(os.walk(train_folder))[1]
# test_ids = next(os.walk(test_folder))[1]
#
# X_train = np.zeros((len(train_ids), image_height, image_width, image_channels), dtype=np.uint8)
# Y_train = np.zeros((len(train_ids), image_height, image_width, 1), dtype=np.bool)

"""Preprocessing the training and the test sample"""

# def preprocess_train(train_ids):
#
#     for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#         path = train_folder + id_
#         img = imread(path + '/images/' + id_ + '.png')[:, :, :image_channels]
#         img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
#         X_train[n] = img
#         mask = np.zeros((image_height, image_width, 1), dtype=np.bool)
#         for mask_file in next(os.walk(path + '/masks/'))[2]:
#             mask_ = imread(path + '/masks/' + mask_file)
#             mask_ = np.expand_dims(resize(mask_, (image_height, image_width), mode='constant', preserve_range=True), axis=-1)
#             mask = np.maximum(mask, mask_)
#         Y_train[n] = mask
#
#     return X_train, Y_train
#
# X_test = np.zeros((len(test_ids), image_height, image_width, image_channels), dtype=np.uint8)
# sizes_test = []
# print('Getting and resizing test images ... ')
# sys.stdout.flush()
#
# def preprocess_test(test_ids):
#
#     for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
#         path = test_folder + id_
#         img = imread(path + '/images/' + id_ + '.png')[:, :, :image_channels]
#         sizes_test.append([img.shape[0], img.shape[1]])
#         img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
#         X_test[n] = img
#
#     return X_test

def combine_generator(gen1, gen2):

    while True:
        yield(gen1.next(), gen2.next())

"""Data Augmentation Routines"""

def train_data_aug(X_data, Y_data, sd, batchsize):

    """Many more arguments can/will be added to the dictionary below"""

    data_gen_args = dict(rotation_range=60.0,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         horizontal_flip=True)

    image_augdata = ImageDataGenerator(**data_gen_args)
    mask_augdata = ImageDataGenerator(**data_gen_args)

    image_augdata.fit(X_data, augment=True, seed=sd)
    mask_augdata.fit(Y_data, augment=True, seed=sd)

    """Use save_to_dir='/home/sinandeger/PycharmProjects/DataScienceBowl18/Aug_img' to output generated images"""

    image_generator = image_augdata.flow(X_data, batch_size=batchsize, seed=sd, shuffle=True)
    mask_generator = mask_augdata.flow(Y_data, batch_size=batchsize, seed=sd, shuffle=True)

    aug_generator = combine_generator(image_generator, mask_generator)

    # aug_generator = zip(aug1, aug2)

    return aug_generator
