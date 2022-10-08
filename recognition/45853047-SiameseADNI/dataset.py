

def load_train_data():
    """ Load image data into tf Dataset, in the form of image pairs
    mapped to labels (0 for same, 1 for different)

    Returns:
        dataset: dataset for train and validation data
    """

    pass

def load_test_data():
    """ Load testing image data, images with labels,
    0 for ad, 1 for cn

    Returns:
        dataset: dataset for testing
    """

    pass

def get_image(path):
    """ Get tf image from path

    Args:
        path (string): path to image

    Returns:
        tf.image: image at path
    """
    pass