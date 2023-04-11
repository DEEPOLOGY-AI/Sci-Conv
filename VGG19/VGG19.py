import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
import os
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers import DepthwiseConv2D, Layer
from tensorflow.keras.preprocessing import image_dataset_from_directory


import keras

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Dropout,ReLU,Layer,Input,add,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2DMaxPool2D,)
from tensorflow.keras import Model

class VGG19Class(Model):
    def __init__(self, classes=1):
        super(VGG19Class, self).__init__()

        self.conv11 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.conv12 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.max1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv21 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")
        self.conv22 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")
        self.max2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv31 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")
        self.conv32 = tf.keras.layers.Conv2D( 256, (3, 3), activation="relu", padding="same")
        self.conv33 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")
        self.conv34 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")
        self.max3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv41 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.conv42 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.conv43 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.conv44 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.max4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.conv51 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.conv52 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.conv53 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.conv54 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.max5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.out = tf.keras.layers.Dense(classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv11(inputs)
        x = self.conv12(x)
        x = self.max1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.max2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)
        x = self.max3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)
        x = self.max4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.conv54(x)
        x = self.max5(x)
        x = self.gap(x)
        x = self.out(x)

        return x
