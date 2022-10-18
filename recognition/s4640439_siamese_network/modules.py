import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, MaxPool2d

IMAGE_SIZE = (240,256,1)
ALPHA = 0.2

"""
Containing the source code of the components of your model. 
Each component must be implementated as a class or a function.
"""

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

def siamese_loss(x0, x1, y: int) -> float:
    """
    Custom loss function for siamese network.

    Takes two vectors, then calculates their distance.

    Vectors of the same class are rewarded for being close and punished for being far away.
    Vectors of different classes are punished for being close and rewarded for being far away.

    Parameters:
        - x0 -- first vector
        - x1 -- second vector
        - y -- integer representing whether or not the two vectors are from the same class

    Returns:
        - loss value
    """
    # TODO
    return 0


def build_binary():
    """
    Generate binary classifier
    This model needs to be a binary classifier that takes an output vector from 
    siamese model and converts it into a single value in the range [0,1]
    """

    # TODO: define layers of model

    model = Sequential()

    return model