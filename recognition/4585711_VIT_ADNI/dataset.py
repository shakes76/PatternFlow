import tensorflow as tf
import keras
from keras.utils import image_dataset_from_directory
from keras import layers

from utils import *

from os.path import exists

def get_data(batch_size, image_size, data_dir):
    if not exists(data_dir + "AD_NC"):
        raise FileNotFoundError(
            "Dataset folder does not exist. Please download data then use unzip.sh.")

    dataset_dir = data_dir + "AD_NC/"
    def get_from_dir(folder):
        return image_dataset_from_directory(dataset_dir + folder,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size)
    
    train_ds = get_from_dir("train")
    test_ds = get_from_dir("test")

    return train_ds, test_ds

def get_normalisation(preprocessing, train_ds, test_ds):
    try:
        with open("normalisation.txt", "r") as f:
            data = f.read().split()
        
        mean = float(data[0])
        var = float(data[1])
    except OSError:
        print("Mean and variance not calculated. Calculating now...")
        ds = train_ds.concatenate(test_ds)
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

def get_data_preprocessing(batch_size=32, image_size=(256, 256), cropped_image_size=(256, 256), cropped_pos=(0, 0), data_dir="./"):
    train_ds, test_ds = get_data(batch_size, image_size, data_dir)

    cropping = ((cropped_pos[0], image_size[0] - (cropped_pos[0] + cropped_image_size[0])),
    (cropped_pos[1], image_size[1] - (cropped_pos[1] + cropped_image_size[1])))

    preprocessing = keras.Sequential([
        layers.Cropping2D(cropping=cropping),
        layers.Rescaling(scale=1./255)
    ])

    preprocessing = get_normalisation(preprocessing, train_ds, test_ds)

    return train_ds, test_ds, preprocessing