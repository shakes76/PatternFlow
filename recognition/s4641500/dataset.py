from modules import *
import glob
import os
from PIL import Image

# load paths
TR_PATH = "keras_png_slices_data/keras_png_slices_train"
TST_PATH = "keras_png_slices_data/keras_png_slices_test"
V_PATH = "keras_png_slices_data/keras_png_slices_validate"

# data dimensions
IMG_H = 80
IMG_W = 80


def load_images(p, image_path):
    """
    Returns a list of resized images at the given path.
    """
    images = []

    for file in image_path:
        image = Image.open(p + '/' + file) 
        image = image.resize((IMG_H, IMG_W))
        image = np.reshape(image, (IMG_H, IMG_W, 1))
        images.append(image)
    return images


def load_dataset():
    train_files = os.listdir(TR_PATH)
    test_files = os.listdir(TST_PATH)
    validation_files = os.listdir(V_PATH) # obsolete?

    # load images from path
    train_imgs = load_images(TR_PATH, train_files)
    test_imgs = load_images(TST_PATH, test_files)
    x_train = np.array(train_imgs)
    x_test = np.array(test_imgs)

    # normalise data to [-0.5, 0.5]
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5

    # get variance for mse
    data_variance = np.var(x_train / 255.0)
    return (train_imgs, test_imgs, x_train_scaled, x_test_scaled, data_variance)