import tensorflow as tf
import tensorflow.keras.layers as layers


class LateralityClassifier:

    def __init__(self, in_shape, use_dropout=False, drop_rate=0.35):
        """
        The constructor for this class, sets self parameters with inputted args
        Args:
            in_shape: the shape of images the classifier must input
            use_dropout: a boolean to decide whether to use dropout layers
            drop_rate: a float value dropout rate to use on dropout layers
        """
        self.in_shape = in_shape
        self.use_dropout = use_dropout
        self.drop_rate = drop_rate

    def build_model(self):
        """
        A method for building the default keras model to use for classification

        Returns:
            The keras Model object
        """
        in_shape = self.in_shape
        droprate = self.drop_rate

        input_layer = layers.Input(shape=in_shape)

        down_samp1 = layers.AvgPool2D((4, 4))(input_layer)

        conv1 = layers.Conv2D(32, (3, 3), padding='same')(down_samp1)
        conv2 = layers.Conv2D(64, (3, 3), padding='same')(conv1)
        pool1 = layers.MaxPooling2D((2, 2))(conv2)

        flatten = layers.Flatten()(pool1)
        drop1 = layers.Dropout(droprate)(flatten)

        if self.use_dropout:
            dense1 = layers.Dense(128, activation="relu")(drop1)
        else:
            dense1 = layers.Dense(128, activation="relu")(flatten)
        drop2 = layers.Dropout(droprate)(dense1)

        if self.use_dropout:
            output = layers.Dense(1, activation="sigmoid")(drop2)
        else:
            output = layers.Dense(1, activation="sigmoid")(dense1)

        model = tf.keras.Model(inputs=input_layer, outputs=output)

        return model

    def build_simple_model(self):
        """
            A method for building an overly simplified keras model to use
            for classification

            Returns:
                The keras Model object
        """
        in_shape = self.in_shape
        input_layer = layers.Input(shape=in_shape)

        flatten = layers.Flatten()(input_layer)

        output = layers.Dense(1, activation="sigmoid")(flatten)
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        return model
