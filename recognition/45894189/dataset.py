import os
import tensorflow
from tensorflow import keras

def load_data(relative_filepath = "/keras_png_slices/"):
    """
    loads image data from the given filepath with normalised values [0, 1]
    param filepath: image set location
    returns: normalised image data
    """
    dirname = os.path.dirname(__file__)
    file_path= os.path.join(dirname, relative_filepath)

    # scale values from [0, 255] to [0, 1]
    image_data = keras.preprocessing.image_dataset_from_directory(file_path, label_mode=None, color_mode="grayscale")
    image_data = image_data.map(lambda x: x / 255.0)
    return image_data