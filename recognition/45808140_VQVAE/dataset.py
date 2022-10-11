import tensorflow as tf
from tensorflow import keras

def load_data(root_path):

    train_data = keras.utils.image_dataset_from_directory('%s/train/' % root_path, labels=None, color_mode='rgb', batch_size=32)
    test_data = keras.utils.image_dataset_from_directory('%s/test/' % root_path, labels=None, color_mode='rgb', batch_size=32)

    #normalise images
    train_data = train_data.map(lambda x: (x / 255.0))
    test_data = test_data.map(lambda x: (x / 255.0))

    return (train_data, test_data)