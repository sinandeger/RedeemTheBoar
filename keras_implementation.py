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
import input_pipeline as inp

from matplotlib.pyplot import imshow

K.set_image_data_format('channels_last')

image_width = 256
image_height = 256
image_channels = 3
train_folder = '../Train/'
#test_folder = '../Test/'

test_folder = '../Stage2_Test/'

train_ids = next(os.walk(train_folder))[1]
test_ids = next(os.walk(test_folder))[1]

X_train = np.zeros((len(train_ids), image_height, image_width, image_channels), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), image_height, image_width, 1), dtype=np.bool)

"""The data input & preprocessing pipeline borrowed from excellent kernels & discussions at the kaggle competition page"""

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = train_folder + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :image_channels]
    img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((image_height, image_width, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (image_height, image_width), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), image_height, image_width, image_channels), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = test_folder + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :image_channels]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
    X_test[n] = img

# for k in range(len(train_ids)):
#
#     """Predictions on the training sample"""
#
#     #rand_img = random.randint(0, len(train_ids)-1)
#     plt.subplot(221)
#     plt.imshow(np.squeeze(X_train[k]))
#     plt.subplot(222)
#     plt.imshow(np.squeeze(Y_train[k]))
#     # plt.subplot(223)
#     # plt.imshow(np.squeeze(preds_train[rand_img]))
#     # plt.subplot(224)
#     # plt.imshow(np.squeeze(preds_train_t[rand_img]))
#     plt.savefig('training_vs_predicted/' + str(train_ids[k]), format='png')
#     plt.show()


"""Uncomment below routine to display images"""

# ix = random.randint(0, len(train_ids))
# imshow(X_train[ix])
# plt.show()
# imshow(np.squeeze(Y_train[ix]))
# plt.show()


"""Test images after data aug applied"""
#
# rand_img = random.randint(0, len(train_ids))
# #imshow(X_train[rand_img])
#
# #img = load_img(X_train[rand_img])  # this is a PIL image
# x = X_train[rand_img]
# x = x.reshape((1,) + x.shape)
#
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in aug_data.flow(x, batch_size=1, save_to_dir='/home/sinandeger/PycharmProjects/DataScienceBowl18',
#                                 save_prefix='aug_data', save_format='png'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely

plot_losses = kerfunc.PlotLosses()


def keras_u_net_model(input_shape):

    X_input = Input(input_shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(X_input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    mp_1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(mp_1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    mp_2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(mp_2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    mp_3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(mp_3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    mp_4 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(mp_4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    output_img = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=X_input, outputs=output_img, name='Keras-U-NetModel')
    return model

#input_img = Input(X_train.shape[1:])
#s = Lambda(lambda x: x / 255)(input_img)

"""Call the Keras U-Net Model"""

Keras_Model = keras_u_net_model(X_train.shape[1:])
Keras_Model.summary()

Keras_Model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])

checkpointer = ModelCheckpoint('/home/sinandeger/PycharmProjects/DataScienceBowl18/Keras_U-Net_Model.h5', verbose=1)
early_stopper = EarlyStopping(monitor='loss', patience=4, verbose=1)

"""Possible Model.fit call back functions: callbacks=[plot_losses, checkpointer, early_stopper]"""

bat_size = 16
seed_no = 1
#aug_count = 10  #Can be used to loop train_data_aug calls

#start = time.time()

train_generator = inp.train_data_aug(X_train, Y_train, seed_no, bat_size)

#end = time.time()

#print start-end, 'seconds'

"""Below call trains the network using the un-augmented data, 0.289 IOU score"""

# Keras_Model.fit(X_train, Y_train, epochs=42, batch_size=32, callbacks=[checkpointer, early_stopper])

"""Below call trains the network using data augmentation, as specified by train_data_aug in input_pipeline.py"""

Keras_Model.fit_generator(train_generator, steps_per_epoch=len(X_train)/bat_size, samples_per_epoch=42, epochs=50,
                          callbacks=[checkpointer])

model = load_model('/home/sinandeger/PycharmProjects/DataScienceBowl18/Keras_U-Net_Model.h5', custom_objects={'mean_iou': kerfunc.mean_iou})

# preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print len(preds_train), len(preds_train_t), len(preds_test), len(preds_test_t)

"""Uncomment below snippet to check predictions on training and test samples. Set iter_counts for how many images """

# iter_count = 1000
#
# for k in range(iter_count):
#
#     """Predictions on the training sample"""
#
#     rand_img = random.randint(0, len(train_ids)-1)
#     plt.subplot(221)
#     plt.imshow(np.squeeze(X_train[rand_img]))
#     plt.subplot(222)
#     plt.imshow(np.squeeze(Y_train[rand_img]))
#     plt.subplot(223)
#     plt.imshow(np.squeeze(preds_train[rand_img]))
#     plt.subplot(224)
#     plt.imshow(np.squeeze(preds_train_t[rand_img]))
#     plt.show()

# for z in range(len(test_ids)):
#
#     plt.subplot(221)
#     plt.imshow(np.squeeze(X_test[z]))
#     plt.subplot(222)
#     plt.imshow(np.squeeze(preds_test[z]))
#     plt.subplot(223)
#     plt.imshow(np.squeeze(preds_test_t[z]))
#     plt.savefig('test_vs_predicted/' + str(test_ids[z]), format='png')


"""Predictions on the test sample"""
#     rand_img = random.randint(0, len(test_ids)-1)
#     plt.subplot(221)
#     plt.imshow(np.squeeze(X_test[rand_img]))
#     plt.subplot(222)
#     plt.imshow(np.squeeze(preds_test[rand_img]))
#     plt.subplot(223)
#     plt.imshow(np.squeeze(preds_test_t[rand_img]))
#     plt.show()

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield kerfunc.rle_encoding(lab_img == i)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('Stage2_valid.csv', index=False)

# for k in range(len(train_ids)):
#
#     """Predictions on the training sample"""
#
#     plt.subplots_adjust(hspace=0.3)
#     plt.subplot(221)
#     plt.title('Training Image')
#     plt.imshow(np.squeeze(X_train[k]))
#     plt.subplot(222)
#     plt.title('Ground Truth Masks')
#     plt.imshow(np.squeeze(Y_train[k]))
#     plt.subplot(223)
#     plt.title('Prediction', fontsize=8)
#     plt.imshow(np.squeeze(preds_train[k]))
#     plt.subplot(224)
#     plt.title('Prediction with p > 0.5', fontsize=8)
#     plt.imshow(np.squeeze(preds_train_t[k]))
#     plt.savefig('training_vs_predicted/' + str(train_ids[k]), format='png')
#     # plt.show()

