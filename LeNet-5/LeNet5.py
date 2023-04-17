import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import (
    Input,  
    Flatten,
    Conv2D,
    AveragePooling2D,
    Dense
)


class LeNetClass(tf.keras.Model):
    def __init__(self, classes=10):
        super(LeNetClass, self).__init__()

        # Convolutional layer 1
        self.conv1 = Conv2D(
            filters=6, kernel_size=(5, 5), activation="tanh"
        )
        # Subsampling layer 1
        self.avg1 = AveragePooling2D(pool_size=(2, 2) , strides=2)

        # Convolutional layer 2
        self.conv2 = Conv2D(
            filters=16, kernel_size=(5, 5), activation="tanh"
        )
        # Subsampling layer 2
        self.avg2 = AveragePooling2D(pool_size=(2, 2) , strides=2)

        # Fully connected layer
        self.flatten = Flatten()
        self.fc1 = Dense(units=120, activation="sigmoid")
        self.fc2 = Dense(units=84, activation="sigmoid")
        self.fc3 = Dense(units=classes, activation="softmax")

    def call(self, inputs):    
        x = self.conv1(inputs)
        x = self.avg1(x)
        x = self.conv2(x)
        x = self.avg2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def LeNet(classes=10 , input_shape=(28, 28, 1,)):

    input_layer = Input(shape=input_shape)
    x = LeNetClass(classes=classes)(input_layer)
    model = Model(inputs=input_layer, outputs=x)

    return model