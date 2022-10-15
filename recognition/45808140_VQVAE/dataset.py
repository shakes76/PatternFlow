import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_data(root_path, batch_size=32):

    train_data = keras.utils.image_dataset_from_directory('%s/train/' % root_path, labels=None, color_mode='grayscale', batch_size=batch_size)
    test_data = keras.utils.image_dataset_from_directory('%s/test/' % root_path, labels=None, color_mode='grayscale', batch_size=batch_size)

    #normalise training data
    normalization_layer = tf.keras.layers.Rescaling(1./255.0)
    train_data = train_data.map(lambda x: normalization_layer(x))
    test_data = test_data.map(lambda x: normalization_layer(x))

    train_variance = np.var(np.concatenate(list(train_data)))

    return (train_data, test_data, train_variance)