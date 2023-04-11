from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Add,
    Reshape,
    DepthwiseConv2D,
    Flatten,
    ReLU,
    GlobalAveragePooling2D
)
from keras.utils.vis_utils import plot_model
from keras import backend as K
import tensorflow as tf

class _conv_block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, strides):
        super(_conv_block, self).__init__()

        self.conv2d1 = Conv2D(filters, kernel, padding="same", strides=strides)
        self.bn1 = BatchNormalization()
        self.act1 = ReLU()

    def call(self, inputs):
        x = self.conv2d1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        return x


class _bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, t, alpha, s, r=False):
        super(_bottleneck, self).__init__()

        self.conv_block2 = _conv_block(filters, (1, 1), (1, 1))
        self.dW_Conv2 = DepthwiseConv2D(
            kernel, strides=(s, s), depth_multiplier=1, padding="same"
        )
        self.bn2 = BatchNormalization()
        self.act2 = ReLU()

        self.conv3 = Conv2D(filters, kernel_size=1, strides=1)
        self.bn3 = BatchNormalization()

    def call(self, inputs):

        x = self.conv_block2(inputs)
        x = self.dW_Conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        return x


class _inverted_residual_block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, t, alpha, strides, n):
        super(_inverted_residual_block, self).__init__()

        self._bottleneck1 = _bottleneck(filters, kernel, t, alpha, strides)
        self.N = n
        self._bottleneck2=[]
        for i in range(n):
            self._bottleneck2.append(_bottleneck(filters, kernel, t, alpha, 1, True) )

    def call(self, inputs):

        x = self._bottleneck1(inputs)
        n_loop = self.N

        for i in range(n_loop):
            print(i)
            x = self._bottleneck2[i](x)

        return x



# Mobilnet
class MobileNetV2Class(tf.keras.Model):
    def __init__(self, classes=10, alpha=1.0):
        super(MobileNetV2Class, self).__init__()

        self.convblock1 = _conv_block(1024, (3, 3), strides=(2, 2))
        self.block2 = _inverted_residual_block(
            16, (3, 3), t=1, alpha=alpha, strides=1, n=1
        )
        self.block3 = _inverted_residual_block(
            24, (3, 3), t=6, alpha=alpha, strides=2, n=1
        )
        self.block3_1 = _inverted_residual_block(
            24, (3, 3), t=6, alpha=alpha, strides=1, n=1
        )
        self.block4 = _inverted_residual_block(
            32, (3, 3), t=6, alpha=alpha, strides=2, n=1
        )
        self.block4_23 = _inverted_residual_block(
            32, (3, 3), t=6, alpha=alpha, strides=1, n=2
        )

        self.block5 = _inverted_residual_block(
            64, (3, 3), t=6, alpha=alpha, strides=2, n=1
        )
        self.block5_234 = _inverted_residual_block(
            64, (3, 3), t=6, alpha=alpha, strides=1, n=3
        )

        self.block6 = _inverted_residual_block(
            96, (3, 3), t=6, alpha=alpha, strides=1, n=3
        )

        self.block7 = _inverted_residual_block(
            160, (3, 3), t=6, alpha=alpha, strides=2, n=1
        )
        self.block7_23 = _inverted_residual_block(
            160, (3, 3), t=6, alpha=alpha, strides=1, n=2
        )
        self.block8 = _inverted_residual_block(
            320, (3, 3), t=6, alpha=alpha, strides=1, n=1
        )

        self.conblock2 = _conv_block(1280, (1, 1), strides=(1, 1))
        self.gap = GlobalAveragePooling2D()
        #self.reshape = Reshape((1, 1, 1280))
        #self.drop = Dropout(0.3, name="Dropout")
        #self.con2d2 = Conv2D(classes, (1, 1), padding="same")
        #self.flat = Flatten()
        self.out = Dense(classes, activation="softmax")

    def call(self, inputs):
        x = self.convblock1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block3_1(x)
        x = self.block4(x)
        x = self.block4_23(x)
        x = self.block5(x)
        x = self.block5_234(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block7_23(x)
        x = self.block8(x)

        x = self.conblock2(x)
        x = self.gap(x)
        #x = self.reshape(x)
        #x = self.drop(x)
        #x = self.con2d2(x)
        #x = self.flat(x)
        x = self.out(x)

        return x


def MobileNetV2(classes=1000):
    input_layer = Input(shape=(224, 224, 3,))
    x = MobileNetV2Class(classes=classes)(input_layer)
    return Model(inputs=input_layer, outputs=x)