import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
from math import sin, cos, sqrt, atan2, radians


import tensorflow as tf


class DataHandler(object):

    def __init__(self,
                 data_params):
        """
        Data parameters initialization
        Args:
            data_params: <dict> containing parameters for data
            batch_size: <int> batch size
        """

        self.data_dir = data_params['data_dir']
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')

    def load_data(self, batch_size, preprocessing):
        """
        Reading and processing data
        Returns: <generators> for train-val data

        """

        if preprocessing == 'inception':
            IMG_HEIGHT, IMG_WIDTH = 299, 299
            # Generator for our training data
            train_image_generator = ImageDataGenerator(rescale=1./255,
                                                       rotation_range=30,
                                                       # zoom_range = 0.3,
                                                       width_shift_range=0.2,
                                                       height_shift_range=0.2,
                                                       horizontal_flip='true')
            # Generator for our validation data
            validation_image_generator = ImageDataGenerator(rescale=1. / 255)
        elif preprocessing == 'resnet':
            IMG_HEIGHT, IMG_WIDTH = 224, 224
            # Generator for our training data
            train_image_generator = ImageDataGenerator(rescale=1./255,
                                                       horizontal_flip='true')
            # Generator for our validation data
            validation_image_generator = ImageDataGenerator(rescale=1. / 255)
        else:
            IMG_HEIGHT, IMG_WIDTH = 224, 224

            train_image_generator = ImageDataGenerator(rescale=1. / 255)
            validation_image_generator = ImageDataGenerator(rescale=1. / 255)

        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                   directory=self.train_dir,
                                                                   shuffle=True,
                                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                   class_mode='categorical')

        val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                      directory=self.val_dir,
                                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                      class_mode='categorical')

        return train_data_gen, val_data_gen
