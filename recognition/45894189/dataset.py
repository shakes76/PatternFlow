import tensorflow
import os
from tensorflow import keras

def load_data(file_path = "recognition\\45894189\keras_png_slices"):
    """
    loads image data from the given filepath with normalised values [0, 1]
    param filepath: image set location
    returns: normalised image data
    """
    file_path = os.path.join(os.getcwd(), file_path)
    image_data = keras.preprocessing.image_dataset_from_directory(file_path, label_mode=None, color_mode="grayscale")
    image_data = image_data.map(lambda x: x / 255.0)
    return image_data