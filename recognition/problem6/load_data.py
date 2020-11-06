import tensorflow as tf



def load_paths(directory):
    """
    Gets all files in the specified directory and returns a sorted list
    of file paths

    Args:
        directory: The directory that we will save the files of
        

    Returns:
        A list of paths ready to be processed.
    """
    # paths will be save to a dictionary to avoid duplications
    training_paths = {}
    for filename in sorted(os.listdir(directory)):
        path = os.path.join(directory, filename)
        training_paths[path] = 1
    return sorted(training_paths.keys())


def load_images(filename):
    """
    Takes the inputted filename and returns the normalised image

    Args:
        file: The filename of the image to be processed
        

    Returns:
        A processed image
    """
    img = tf.io.read_file(filename)
    # convert to 3 channel
    img = tf.io.decode_png(img, channels=3)
    # cast to float to allowed easier manipulation
    img = tf.dtypes.cast(img, tf.float32)
    # normalise between [0,1]
    img = img / 255.0
    return img
