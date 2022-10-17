import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, MaxPool2d

IMAGE_SIZE = (240,256,1)
ALPHA = 0.2

def build_siamese():
    """
    Generate Siamese model
    This model needs to be a CNN that reduces an image to a vector
    """ 
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, input_shape=IMAGE_SIZE))
    model.add(LeakyReLU(alpha=ALPHA))

    model.add(MaxPool2d(pool_size=(2,2), strides=(1, 1)))

    model.add(Conv2D(64, kernel_size=3))
    model.add(LeakyReLU(alpha=ALPHA))

    model.add(MaxPool2d(pool_size=(2,2), strides=(1, 1)))

    model.add(Conv2D(128, kernel_size=3))
    model.add(LeakyReLU(alpha=ALPHA))

    model.add(MaxPool2d(pool_size=(2,2), strides=(1, 1)))

    model.add(Flatten())

    return model



def build_binary():
    """
    Generate binary classifier
    This model needs to be a binary classifier that takes an output vector from 
    siamese model and converts it into a single value in the range [0,1]
    """

    # TODO: define layers of model

    model = Sequential()

    return model