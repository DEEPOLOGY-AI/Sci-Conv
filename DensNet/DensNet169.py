import tensorflow as tf
import keras
import tensorflow as tf
from keras.models import Model



class Conv_block(tf.keras.layers.Layer):
    def __init__(self, growth_rate):
        super(Conv_block, self).__init__()
        self.growth_rate = growth_rate
        self.bn_axis = 3

        self.bn1 = tf.keras.layers.BatchNormalization(
            axis=self.bn_axis, epsilon=1.001e-5,
        )
        self.act1 = tf.keras.layers.Activation("relu")
        self.conv1 = tf.keras.layers.Conv2D(4 * self.growth_rate, 1, use_bias=False,)

        self.bn2 = tf.keras.layers.BatchNormalization(
            axis=self.bn_axis, epsilon=1.001e-5,
        )
        self.act2 = tf.keras.layers.Activation("relu")
        self.conv2 = tf.keras.layers.Conv2D(
            self.growth_rate, 3, padding="same", use_bias=False,
        )

        self.concatenate = tf.keras.layers.Concatenate(axis=self.bn_axis)

    def call(self, inputs):
        x1 = inputs
        x = self.bn1(inputs)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)

        return self.concatenate([x1, x])


class Dense_block(tf.keras.layers.Layer):
    def __init__(self, blocks):
        super(Dense_block, self).__init__()
        self.blocks = blocks
      
        self.conv_dict = []
        for i in range(self.blocks):
            self.conv_dict.append(Conv_block(32)) 


    def call(self, inputs):
        x = inputs
        for i in range(self.blocks):
            x = self.conv_dict[i](x)
        return x


class Transition_block(tf.keras.layers.Layer):
    def __init__(self, reduction,x_shape):
        super(Transition_block, self).__init__()
        self.reduction = reduction
        self.bn_axis = 3
        self.bn1 = tf.keras.layers.BatchNormalization(
            axis=self.bn_axis, epsilon=1.001e-5
        )
        self.act1 = tf.keras.layers.Activation("relu")
        self.conv1=tf.keras.layers.Conv2D( x_shape* reduction, 1,use_bias=False,)
        self.avg = tf.keras.layers.AveragePooling2D(2, strides=2)

    def call(self, inputs):
        x = self.bn1(inputs)
        x = self.act1(x)
        x=self.conv1(x)
 
        x=self.avg(x)
        return x


class DenseNet169Class(tf.keras.models.Model):
    def __init__(self, classes=10):
        super(DenseNet169Class, self).__init__()
        self.bn_axis = 3
        self.classes = classes

        self.blocks = [6, 12, 32, 32]

        self.zeropad1 = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))
        self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False,)
        self.bn1 = tf.keras.layers.BatchNormalization(
            axis=self.bn_axis, epsilon=1.001e-5
        )
        self.act1 = tf.keras.layers.Activation("relu")
        self.zeropad2 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))
        self.maxp1 = tf.keras.layers.MaxPooling2D(3, strides=2,)

        self.dense_block1 = Dense_block(self.blocks[0])
        self.transition_block1 = Transition_block(0.5,256)
        self.dense_block2 = Dense_block(self.blocks[1])
        self.transition_block2 = Transition_block(0.5,512)
        self.dense_block3 = Dense_block(self.blocks[2])
        self.transition_block3 = Transition_block(0.5,1280)
        self.dense_block4 = Dense_block(self.blocks[3])

        self.bn2 = tf.keras.layers.BatchNormalization(
            axis=self.bn_axis, epsilon=1.001e-5
        )
        self.act2 = tf.keras.layers.Activation("relu")
        self.avg = tf.keras.layers.GlobalAveragePooling2D()
        self.out = tf.keras.layers.Dense(self.classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.zeropad1(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.zeropad2(x)
        x = self.maxp1(x)

        x = self.dense_block1(x)
        x = self.transition_block1(x)
        x = self.dense_block2(x)
        x = self.transition_block2(x)
        x = self.dense_block3(x)
        x = self.transition_block3(x)
        x = self.dense_block4(x)

        x = self.bn2(x)
        x = self.act2(x)
        x = self.avg(x)
        x = self.out(x)

        return x


def DenseNet169(classes=10,input_shape=(128, 128, 3,)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = DenseNet169Class(classes=classes)(inputs)
    return Model(inputs=inputs, outputs=x)