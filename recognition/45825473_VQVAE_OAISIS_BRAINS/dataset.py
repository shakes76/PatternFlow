from google.colab import drive
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from PIL import Image
import zipfile
import matplotlib.pyplot as plt
import numpy as np


def get_image_slices():
    googleCollab = False
    if (googleCollab):
        drive.mount('/content/gdrive', force_remount=True)
        zipped_images = zipfile.ZipFile('/content/gdrive/MyDrive/Colab Notebooks/OASISProcessed.zip')
    else:
        zipped_images = zipfile.ZipFile('./OASISProcessed.zip')
    images = []
    zipped_images.extractall()

    parent_dir = "./keras_png_slices_data"
    train_path = parent_dir + "/keras_png_slices_train"
    test_path = parent_dir + "/keras_png_slices_test"
    val_path = parent_dir + "/keras_png_slices_validate"
    train_images = []
    test_images = []
    validate_images = []

    for index, path in enumerate([train_path, test_path, val_path]):
        for image in listdir(path):
            if isfile(join(path, image)):
                img = tf.io.read_file(join(path, image))
                img = tf.image.decode_png(img, channels=1)
                img = tf.image.resize(img, (128, 128))
                img = tf.cast(img, tf.float32)
                img = tf.squeeze(img)
                if index == 0:
                    train_images.append(img)
                elif index == 1:
                    test_images.append(img)
                else:
                    validate_images.append(img)

    # This original approach to was deprecated as it needed to be expanded 
    # in order to downsize the image from 256x256 to 128x128 - I was running out of memory allocating 
    # space for tensors in the latent space for all 9664 training images the original images were 256x256 in size.
    
    # train_images = [np.array(Image.open(join(train_path, f))) for f in listdir(train_path) if isfile(join(train_path, f))]
    # test_images = [np.array(Image.open(join(test_path, f))) for f in listdir(test_path) if isfile(join(test_path, f))]
    # validate_images = [np.array(Image.open(join(val_path, f))) for f in listdir(val_path) if isfile(join(val_path, f))]
    return train_images, test_images, validate_images

