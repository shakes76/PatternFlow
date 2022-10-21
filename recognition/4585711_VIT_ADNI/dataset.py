import tensorflow as tf
import keras
from keras.utils import image_dataset_from_directory
from keras import layers

from os.path import exists

def get_data(batch_size, image_size, data_dir):
    if not exists(data_dir + "AD_NC"):
        raise FileNotFoundError(
            "Dataset folder does not exist. Please download data then use unzip.sh.")

    dataset_dir = data_dir + "AD_NC/"
    def get_from_dir(folder):
        ds = image_dataset_from_directory(dataset_dir + folder,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size)

        ds.prefetch(tf.data.AUTOTUNE)

        return ds
    
    train_ds = get_from_dir("train")
    test_ds = get_from_dir("test")
    valid_ds = get_from_dir("valid")

    return train_ds, test_ds, valid_ds

class TrackCrop(layers.Layer):
    def __init__(self, cropped_image_size):
        super(TrackCrop, self).__init__()

        self.cropped_image_size = cropped_image_size

    def crop(self, item):
        image, offset_height, offset_width = item
        return tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, self.cropped_image_size[0], self.cropped_image_size[1]
        ), offset_height, offset_width

    def call(self, inputs):
        _, size_y, size_x, _ = inputs.shape

        non_black = tf.cast(inputs != 0.0, tf.int32)
        count = tf.reduce_sum(non_black, axis=[1,2,3])

        X, Y = tf.meshgrid(range(size_x), range(size_y))
        X = tf.reshape(X, (size_y, size_x, 1))
        Y = tf.reshape(Y, (size_y, size_x, 1))

        x = tf.cast(tf.reduce_sum(X * non_black, axis=[1,2,3]) / count, tf.int32)
        y = tf.cast(tf.reduce_sum(Y * non_black, axis=[1,2,3]) / count, tf.int32)

        H = tf.math.maximum(y - self.cropped_image_size[0]//2, 0)
        W = tf.math.maximum(x - self.cropped_image_size[1]//2, 0)

        H = tf.math.minimum(H, size_y - 1 - self.cropped_image_size[0])
        W = tf.math.minimum(W, size_x - 1 - self.cropped_image_size[1])

        return tf.map_fn(self.crop, (inputs, H, W))[0]

def get_normalisation(preprocessing, train_ds, test_ds, valid_ds):
    try:
        with open("normalisation.txt", "r") as f:
            data = f.read().split()
        
        mean = float(data[0])
        var = float(data[1])
    except OSError:
        print("Mean and variance not calculated. Calculating now...")
        ds = train_ds.concatenate(test_ds).concatenate(valid_ds)
        ds = ds.map(lambda input,tag : input)
        ds = ds.map(lambda item : preprocessing(item))
        normalise = layers.Normalization(axis=None)
        normalise.adapt(ds)

        mean = normalise.adapt_mean.numpy()
        var = normalise.adapt_variance.numpy()
        with open("normalisation.txt", "w") as f:
            f.write(str(mean) + " " + str(var))

    return keras.Sequential([
        preprocessing,
        layers.Normalization(axis=None, mean=mean, variance=var)
    ])

def get_data_preprocessing(batch_size=32, image_size=(256, 256), cropped_image_size=(256, 256), data_dir="./"):
    train_ds, test_ds, valid_ds = get_data(batch_size, image_size, data_dir)

    preprocessing = keras.Sequential([
        #TrackCrop(cropped_image_size),
        layers.Cropping2D((8, 0)),
        layers.Rescaling(scale=1./255)
    ])

    preprocessing = get_normalisation(preprocessing, train_ds, test_ds, valid_ds)

    """preprocessing = keras.Sequential([
        preprocessing,
        layers.RandomTranslation(0.0, (-0.2,0.0), fill_mode='constant'),
        layers.RandomZoom((-0.05, 0.1), fill_mode='constant')
    ])"""

    return train_ds, test_ds, valid_ds, preprocessing