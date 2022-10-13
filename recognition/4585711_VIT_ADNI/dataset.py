import tensorflow as tf
import keras
from keras.utils import image_dataset_from_directory
from keras import layers

from os.path import exists

DATA_DIR = "/data/s4585711/vit/"

def get_data(batch_size, image_size):
    if not exists(DATA_DIR + "AD_NC"):
        raise FileNotFoundError(
            "Dataset folder does not exist. Please download data then use unzip.sh.")

    dataset_dir = DATA_DIR + "AD_NC/"
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

def get_data_preprocessing(batch_size=32, image_size=(256, 256), cropped_image_size=(256, 256), cropped_pos=(0, 0)):
    train_ds, test_ds = get_data(batch_size, image_size)

    cropping = ((cropped_pos[0], image_size[0] - (cropped_pos[0] + cropped_image_size[0])),
    (cropped_pos[1], image_size[1] - (cropped_pos[1] + cropped_image_size[1])))

    preprocessing = keras.Sequential([
        layers.Cropping2D(input_shape=(image_size[0], image_size[1], 1), cropping=cropping),
        layers.Rescaling(scale=1./255)
    ])

    preprocessing = get_normalisation(preprocessing, train_ds, test_ds)

    return train_ds.map(lambda x,y : (preprocessing(x), y)), test_ds.map(lambda x,y : (preprocessing(x), y)), preprocessing

if __name__ == "__main__":
    train_ds, test_ds, preprocessing = get_data_preprocessing(image_size=(240, 256), cropped_image_size=(192, 160), cropped_pos=(20, 36))

    preprocessing.summary(expand_nested=True)