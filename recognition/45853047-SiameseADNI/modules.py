from keras.models import Model

import keras as k
import tensorflow as tf
import keras.layers as kl
import keras.backend as kb
import tensorflow_addons as tfa



def subnetwork(height, width):
    """ The identical subnetwork in the SNN

    Returns:
        tf.keras.Model: the subnetwork Model
    """
    # may need dropout and batch norm
    subnet = k.Sequential(layers=[
            kl.Flatten(input_shape=(height, width, 1)),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dropout(0.3),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2')
        ], name='subnet'
    )

    # subnet = k.Sequential(layers=[
    #         kl.Conv2D(64, (2, 2), input_shape=(height, width, 1)),
    #         kl.MaxPooling2D(),
    #         kl.Dropout(0.3),
    #         kl.Flatten(),
    #         kl.Dense(512, activation='relu',kernel_regularizer='l2'),
    #         kl.Dense(256, activation='relu',kernel_regularizer='l2'),
    #     ], name='subnet'
    # )



    return subnet

# TODO: is this euclidean?
def distance_layer(im1_feature, im2_feature):
    """ Layer to compute (euclidean) difference between feature vectors

    Args:
        im1_feature (tensor): feature vector of an image
        im2_feature (tensor): feature vector of an image

    Returns:
        tensor: Tensor containing differences
    """
    tensor = kb.sum(kb.square(im1_feature - im2_feature), axis=1, keepdims=True)
    return kb.sqrt(kb.maximum(tensor, kb.epsilon())) 

def contrastive_loss(y, y_pred):

    square = tf.math.square(y_pred)
    margin = tf.math.square(tf.math.maximum(1 - (y_pred), 0))
    return tf.math.reduce_mean((1 - y) * square + (y) * margin
    )



def siamese(height: int, width: int):
    """ The SNN. Passes image pairs through the subnetwork,
        and computes distance between output vectors. 

    Args:
        height (int): height of input image
        width (int): width of input imagee

    Returns:
        Model: compiled model
    """

    subnet = subnetwork(height, width)

    image1 = kl.Input((height, width, 1))
    image2 = kl.Input((height, width, 1))

    feature1 = subnet(image1)
    feature2 = subnet(image2)

    distance = distance_layer(feature1, feature2)

    # Classification
    tensor = kl.BatchNormalization()(distance)
    out = kl.Dense(units = 1, activation='sigmoid')(tensor)

    model = Model([image1, image2], out)

    # TODO: may not need decay
    # opt = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay= 0.01) # could be .99
    opt = tf.optimizers.Adam(learning_rate=0.0001)

    # TODO: change loss to contrastive
    model.compile(loss=contrastive_loss, metrics=['accuracy'],optimizer=opt)

    return model

def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss