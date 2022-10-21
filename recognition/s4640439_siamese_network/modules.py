import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense

IMAGE_SIZE = (240,256,1)
ALPHA = 0.2

SIAMESE_OUTPUT_SHAPE = (512,)

def build_siamese():
    """
    Generate Siamese model
    This model needs to be a CNN that reduces an image to a vector
    """ 
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=IMAGE_SIZE, 
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(LeakyReLU(alpha=0.2))

    return model

def build_binary():
    """
    Generate binary classifier
    This model needs to be a binary classifier that takes an output embedding from 
    siamese model and converts it into a single value in the range [0,1] for classification
    """
    model = Sequential()

    model.add(Dense(64, input_shape=SIAMESE_OUTPUT_SHAPE, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model