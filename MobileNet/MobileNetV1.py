import tensorflow as tf
import keras

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    DepthwiseConv2D,
    BatchNormalization,
    ReLU,
    AvgPool2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras import Model
from keras.layers import (
    Dropout,
    Input,
    Add,
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
from keras.initializers import glorot_uniform


# Mobilenet Bolk
class mobilnet_block(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(mobilnet_block, self).__init__()

        self.dW_Conv = DepthwiseConv2D(kernel_size=3, strides=strides, padding="same")
        self.bn1 = BatchNormalization()
        self.act1 = ReLU()

        self.conv2 = Conv2D(filters=filters, kernel_size=1, strides=1)
        self.bn2 = BatchNormalization()
        self.act2 = ReLU()

    def call(self, inputs):

        x = self.dW_Conv(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x


# Mobilnet
class MobileNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()

        # Streaming of Bulding Mobilenet
        self.conv3 = Conv2D(filters=32, kernel_size=3, strides=2, padding="same")
        self.bn3 = BatchNormalization()
        self.act3 = ReLU()

        # Body of the Model

        self.block1 = mobilnet_block(filters=64, strides=1)
        self.block2 = mobilnet_block(filters=128, strides=2)
        self.block3 = mobilnet_block(filters=128, strides=1)
        self.block4 = mobilnet_block(filters=256, strides=2)
        self.block5 = mobilnet_block(filters=256, strides=1)
        self.block6 = mobilnet_block(filters=512, strides=2)
        self.block7to11 = []

        for i in range(5):
            self.block7to11.append(mobilnet_block(filters=512, strides=1))

        self.block12 = mobilnet_block(filters=1024, strides=2)
        self.block13 = mobilnet_block(filters=1024, strides=2)

        # End Model
        self.globalavg = GlobalAveragePooling2D()
        self.out = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv3(inputs)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        for i in range(5):
            x = self.block7to11[i](x)

        x = self.block12(x)
        x = self.block13(x)

        x = self.globalavg(x)
        x = self.out(x)

        return x


def MobileNetV1(classes=10,input_shape=(128, 128, 3,)):
    input_layer = Input(shape=input_shape)
    x = MobileNet(num_classes=classes)(input_layer)
    return Model(inputs=input_layer, outputs=x)