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
from tensorflow.keras.layers import (
    Dropout,
    ReLU,
    Layer,
    Input,
    add,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    GlobalMaxPooling2D,
    MaxPool2D,
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform

# Model Checks
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle

# Model Graphs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from keras.layers import Activation, Lambda, GlobalAveragePooling2D, concatenate
import math
from tensorflow.keras.layers import DepthwiseConv2D
from keras.layers import SeparableConv2D, Add, GlobalAvgPool2D
from keras.layers import Resizing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


class conv2d_bn(tf.keras.layers.Layer):
    def __init__(self, f, w, h, padding="same", strides=(1, 1)):
        super(conv2d_bn, self).__init__()
        self.conv = Conv2D(f, (w, h), strides=strides, padding=padding)
        self.bn = BatchNormalization(axis=3, scale=False)
        self.act = Activation("relu")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x


############################################################################


class inception_block_a(tf.keras.layers.Layer):
    def __init__(self, conv_block=None):
        super(inception_block_a, self).__init__()
        if conv_block is None:
            conv_block = conv2d_bn
        self.branch1x1 = conv_block(64, 1, 1)

        self.branch5x5_1 = conv_block(48, 1, 1)
        self.branch5x5_2 = conv_block(64, 5, 5)

        self.branch3x3dbl_1 = conv_block(64, 1, 1)
        self.branch3x3dbl_2 = conv_block(96, 3, 3)
        self.branch3x3dbl_3 = conv_block(96, 3, 3)

        self.branch_pool_2D = AveragePooling2D((3, 3), strides=(1, 1), padding="same")
        self.branch_pool = conv_block(32, 1, 1)

    def call(self, inputs):
        branch1x1 = self.branch1x1(inputs)

        branch5x5 = self.branch5x5_1(inputs)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(inputs)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool_2D(inputs)
        branch_pool = self.branch_pool(branch_pool)

        outputs = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)
        return outputs


############################################################################


class reduction_block_a(tf.keras.layers.Layer):
    def __init__(self, conv_block=None):
        super(reduction_block_a, self).__init__()
        if conv_block is None:
            conv_block = conv2d_bn
        self.branch3x3 = conv_block(384, 3, 3, strides=(2, 2), padding="valid")

        self.branch3x3dbl_1 = conv_block(64, 1, 1)
        self.branch3x3dbl_2 = conv_block(96, 3, 3)
        self.branch3x3dbl_3 = conv_block(96, 3, 3, strides=(2, 2), padding="valid")

        self.branch_pool = AveragePooling2D((3, 3), strides=(2, 2))

    def call(self, inputs):
        branch3x3 = self.branch3x3(inputs)

        branch3x3dbl = self.branch3x3dbl_1(inputs)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(inputs)

        outputs = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3)
        return outputs


############################################################################


class inception_block_b(tf.keras.layers.Layer):
    def __init__(self, conv_block=None):
        super(inception_block_b, self).__init__()
        if conv_block is None:
            conv_block = conv2d_bn
        self.branch1x1 = conv_block(192, 1, 1)

        self.branch7x7_1 = conv_block(128, 1, 1)
        self.branch7x7_2 = conv_block(128, 1, 7)
        self.branch7x7_3 = conv_block(192, 7, 1)

        self.branch7x7dbl_1 = conv_block(128, 1, 1)
        self.branch7x7dbl_2 = conv_block(128, 7, 1)
        self.branch7x7dbl_3 = conv_block(128, 1, 7)
        self.branch7x7dbl_4 = conv_block(128, 7, 1)
        self.branch7x7dbl_5 = conv_block(128, 1, 7)

        self.branch_pool_2D = AveragePooling2D((3, 3), strides=(1, 1), padding="same")
        self.branch_pool = conv_block(192, 1, 1)

    def call(self, inputs):
        branch1x1 = self.branch1x1(inputs)

        branch7x7 = self.branch7x7_1(inputs)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(inputs)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool_2D(inputs)
        branch_pool = self.branch_pool(branch_pool)

        outputs = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)
        return outputs


############################################################################


class reduction_block_b(tf.keras.layers.Layer):
    def __init__(self, conv_block=None):
        super(reduction_block_b, self).__init__()
        if conv_block is None:
            conv_block = conv2d_bn

        self.branch3x3_1 = conv_block(192, 1, 1)
        self.branch3x3_2 = conv_block(320, 3, 3, strides=(2, 2), padding="valid")

        self.branch7x7x3_1 = conv_block(192, 1, 1)
        self.branch7x7x3_2 = conv_block(192, 1, 7)
        self.branch7x7x3_3 = conv_block(192, 7, 1)
        self.branch7x7x3_4 = conv_block(192, 3, 3, strides=(2, 2), padding="valid")

        self.branch_pool = AveragePooling2D((3, 3), strides=(2, 2))

    def call(self, inputs):
        branch3x3 = self.branch3x3_1(inputs)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(inputs)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.branch_pool(inputs)

        outputs = concatenate([branch3x3, branch7x7x3, branch_pool], axis=3)
        return outputs


############################################################################


class inception_block_c(tf.keras.layers.Layer):
    def __init__(self, conv_block=None):
        super(inception_block_c, self).__init__()
        if conv_block is None:
            conv_block = conv2d_bn

        self.branch1x1 = conv_block(320, 1, 1)

        self.branch3x3 = conv_block(384, 1, 1)
        self.branch3x3_1 = conv_block(384, 1, 3)
        self.branch3x3_2 = conv_block(384, 3, 1)

        self.branch3x3dbl_01 = conv_block(448, 1, 1)
        self.branch3x3dbl_02 = conv_block(384, 3, 3)
        self.branch3x3dbl_1 = conv_block(384, 1, 3)
        self.branch3x3dbl_2 = conv_block(384, 3, 1)

        self.branch_pool_2D = AveragePooling2D((3, 3), strides=(1, 1), padding="same")
        self.branch_pool = conv_block(192, 1, 1)

    def call(self, inputs):
        branch1x1 = self.branch1x1(inputs)

        branch3x3 = self.branch3x3(inputs)
        branch3x3_1 = self.branch3x3_1(branch3x3)
        branch3x3_2 = self.branch3x3_2(branch3x3)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=3)

        branch3x3dbl = self.branch3x3dbl_01(inputs)
        branch3x3dbl = self.branch3x3dbl_02(branch3x3dbl)
        branch3x3dbl_1 = self.branch3x3dbl_1(branch3x3dbl)
        branch3x3dbl_2 = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = self.branch_pool_2D(inputs)
        branch_pool = self.branch_pool(branch_pool)

        outputs = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3)
        return outputs


############################################################################


class InceptionV3Class(Model):
    def __init__(self, classes=1000, conv_block=None):
        super(InceptionV3Class, self).__init__()
        if conv_block is None:
            conv_block = conv2d_bn
        self.conv0 = conv_block(32, 3, 3, strides=(2, 2), padding="valid")
        self.conv1 = conv_block(32, 3, 3, padding="valid")
        self.conv2 = conv_block(64, 3, 3)
        self.max_pool_2D_0 = MaxPooling2D((3, 3), strides=(2, 2))

        self.conv3 = conv_block(80, 1, 1, padding="valid")
        self.conv4 = conv_block(192, 3, 3, padding="valid")
        self.max_pool_2D_1 = MaxPooling2D((3, 3), strides=(2, 2))

        self.inc_a0 = inception_block_a()
        self.inc_a1 = inception_block_a()
        self.inc_a2 = inception_block_a()
        self.rec_a = reduction_block_a()
        self.inc_b0 = inception_block_b()
        self.inc_b1 = inception_block_b()
        self.inc_b2 = inception_block_b()
        self.inc_b3 = inception_block_b()
        self.rec_b = reduction_block_b()
        self.inc_c0 = inception_block_c()
        self.inc_c1 = inception_block_c()

        self.gap_pool_2D_1 = GlobalAveragePooling2D()
        self.dense_0 = Dense(1000, activation="relu")
        self.dense_1 = Dense(512, activation="relu")
        self.drop = Dropout(0.2)
        self.dense_2 = Dense(classes, activation="softmax")

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool_2D_0(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.inc_a0(x)
        x = self.inc_a1(x)
        x = self.inc_a2(x)
        x = self.rec_a(x)
        x = self.inc_b0(x)
        x = self.inc_b1(x)
        x = self.inc_b2(x)
        x = self.inc_b3(x)
        x = self.rec_b(x)
        x = self.inc_c0(x)
        x = self.inc_c1(x)
        x = self.gap_pool_2D_1(x)
        x = self.dense_0(x)
        x = self.dense_1(x)
        x = self.drop(x)
        x = self.dense_2(x)

        return x


############################################################################

def InceptionV3(classes=1000 , input_shape=(128, 128, 3,)):
    channel_axis = 3
    input_layer = Input(input_shape)
    x = InceptionV3Class(classes=classes)(input_layer)

    return Model(inputs=input_layer, outputs=x)