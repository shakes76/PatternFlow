



def subnetwork():
    """ The identical subnetwork in the SNN

    Returns:
        tf.keras.Model: the subnetwork Model
    """
    pass

# TODO: is this euclidean?
def distance_layer(im1_feature, im2_feature):
    """ Layer to compute (euclidean) difference between feature vectors

    Args:
        im1_feature (tensor): feature vector of an image
        im2_feature (tensor): feature vector of an image

    Returns:
        tensor: Tensor containing differences
    """
    pass


def siamese():
    """ The SNN. Passes image pairs through the subnetwork,
        and computes distance between output vectors. 

    Returns:
        Model: compiled model
    """
    pass