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
    # subnet = k.Sequential(layers=[
    #         kl.Flatten(input_shape=(height, width, 1)),
    #         kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
    #         kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
    #         kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
    #         kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
    #         kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
    #         kl.Dense(1024, activation='relu',kernel_regularizer='l2')
    #     ], name='subnet'
    # )

    subnet = k.Sequential(layers=[
            kl.Conv2D(64, (2, 2), input_shape=(height, width, 1)),
            kl.MaxPooling2D(),
            kl.Dropout(0.3),
            kl.Flatten(),
            kl.Dense(512, activation='relu',kernel_regularizer='l2'),
            kl.Dense(256, activation='relu',kernel_regularizer='l2'),
        ], name='subnet'
    )



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

def contrastive_loss():
    pass


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
    out = kl.Dense(units = 1, activation='sigmoid')(distance)

    model = Model([image1, image2], out)

    # TODO: may not need decay
    opt = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay= 0.01) # could be .99

    # TODO: change loss to contrastive
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],optimizer=opt)

    return model