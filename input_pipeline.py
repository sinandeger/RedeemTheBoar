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

def combine_generator(gen1, gen2):

    while True:
        yield(gen1.next(), gen2.next())

"""Data Augmentation Routines"""

def train_data_aug(X_data, Y_data, sd, batchsize):

    """Many more arguments can/will be added to the dictionary below"""

    data_gen_args = dict(rotation_range=90.0,
                         width_shift_range=0.25,
                         height_shift_range=0.25,
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
