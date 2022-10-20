from modules import *
import glob
import os
from PIL import Image

# find and initialise dataset
train_path = "keras_png_slices_data/keras_png_slices_train"
test_path = "keras_png_slices_data/keras_png_slices_test"
validation_path = "keras_png_slices_data/keras_png_slices_validate"
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
validation_files = os.listdir(validation_path)

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

train_imgs = load_images(train_path, train_files)
test_imgs = load_images(test_path, test_files)
x_train = np.array(train_imgs)
x_test = np.array(test_imgs)

# normalise data to [-0.5, 0.5]
x_train_scaled = (x_train / 255.0) - 0.5
x_test_scaled = (x_test / 255.0) - 0.5

# get variance for mse
data_variance = np.var(x_train / 255.0)

# Check shapes of arrays
print(x_train.shape)
print(data_variance.shape)