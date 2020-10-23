import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2


class LateralityClassifier:

    def __init__(self, in_shape, use_dropout=False, drop_rate=0.35,
                 use_l2=False, l2_param=0.001):
        self.in_shape = in_shape
        self.use_dropout = use_dropout
        self.drop_rate = drop_rate
        self.use_l2 = use_l2
        self.l2_param = l2_param

    def build_model(self):
        in_shape = self.in_shape
        droprate = self.drop_rate
        l2_param = self.l2_param

        input_layer = layers.Input(shape=in_shape)

        down_samp1 = layers.AvgPool2D((4, 4))(input_layer)

        conv1 = layers.Conv2D(8, (4, 4), padding='same', kernel_regularizer=l2(l2_param))(down_samp1)
        conv2 = layers.Conv2D(16, (4, 4), padding='same', kernel_regularizer=l2(l2_param))(conv1)
        pool1 = layers.MaxPooling2D((2, 2))(conv2)

        conv3 = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_param))(pool1)
        conv4 = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_param))(conv3)
        pool2 = layers.MaxPooling2D((2, 2))(conv4)

        conv5 = layers.Conv2D(128, (2, 2), padding='same', kernel_regularizer=l2(l2_param))(pool2)
        conv6 = layers.Conv2D(256, (2, 2), padding='same', kernel_regularizer=l2(l2_param))(conv5)
        pool3 = layers.MaxPooling2D((2, 2))(conv6)

        flatten = layers.Flatten()(pool2)

        dropbc1 = layers.Dropout(droprate)(flatten)
        dense0 = layers.Dense(256, activation="relu")(dropbc1)

        drop0 = layers.Dropout(droprate)(dense0)
        dense1 = layers.Dense(128, activation="relu")(drop0)

        drop1 = layers.Dropout(droprate)(dense1)
        # dense2 = layers.Dense(128, activation = "relu") (drop1)

        # drop2 = layers.Dropout(droprate) (dense2)
        dense3 = layers.Dense(64, activation="relu")(drop1)

        drop3 = layers.Dropout(droprate)(dense3)
        dense4 = layers.Dense(32, activation="relu")(drop3)

        output = layers.Dense(1, activation="sigmoid")(dense4)
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        return model

    def build_simple_model(self):
        in_shape = self.in_shape
        input_layer = layers.Input(shape=in_shape)

        flatten = layers.Flatten()(input_layer)

        output = layers.Dense(1, activation="sigmoid")(flatten)
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        return model
