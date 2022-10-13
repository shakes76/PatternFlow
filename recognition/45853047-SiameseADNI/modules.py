from keras.models import Model
import tensorflow as tf
import tensorflow.keras as k
import keras.layers as kl
import keras.backend as kb



def subnetwork(height, width):
    """ The identical subnetwork in the SNN

    Returns:
        tf.keras.Model: the subnetwork Model
    """
    subnet = k.Sequential(layers=[
            kl.Flatten(input_shape=(height, width, 1)),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
            kl.Dense(1024, activation='relu',kernel_regularizer='l2'),
        ], name='subnet'
    )

    return subnet


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


def classification_model(subnet):
    """ Build the classification Model

    Args:
        subnet (layer): the sequential layer trained in the SNN

    Returns:
        model: compiled model
    """
    image = kl.Input((128, 128, 1))
    tensor = subnet(image)
    tensor = kl.BatchNormalization()(tensor)
    out = kl.Dense(units = 1, activation='sigmoid')(tensor)

    classifier = Model([image], out)

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    classifier.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)

    return classifier

def contrastive_loss(y, y_pred):
    """
    Sourced from [Siameses Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive/)
    """
    square = tf.math.square(y_pred)
    margin = tf.math.square(tf.math.maximum(1 - (y_pred), 0))
    return tf.math.reduce_mean((1 - y) * square + (y) * margin)



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

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss=contrastive_loss, metrics=['accuracy'],optimizer=opt)

    return model