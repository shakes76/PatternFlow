import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_datasets as tfds
from model import improv_unet

# download pictures
tf.keras.utils.\
    get_file(origin="https://cloudstor.aarnet.edu.au/sender/?s=download&token=723595dd-15b0-4d1e-87b8-237a7fe282ff",
             fname=os.getcwd() + '\ISIC2018_Task1-2_Training_Data.zip', extract=True, cache_dir=os.getcwd())

# constants
IMAGE_H = IMAGE_W = 180
NUM_EXAMPLES = 3
TRAIN_TEST_SPLIT = 0.8
BAT_SIZE = 32
CHANNEL_NUM = 3


def convert_images(f_path, mask):
    """
    convert images for code use
    Args:
        f_path: file path of images
        mask: is image a mask

    Returns: converted images

    """
    if mask:
        image = tf.image.decode_png(tf.io.read_file(f_path), channels=1)
        image_converted = tf.image.convert_image_dtype(image, tf.float32)
        image_resized = tf.cast(tf.image.resize(image_converted, size=(IMAGE_H, IMAGE_W)), tf.float32) / 255.0
    else:
        image = tf.image.decode_jpeg(tf.io.read_file(f_path), channels=3)
        image_converted = tf.image.convert_image_dtype(image, tf.float32)
        image_resized = tf.image.resize(image_converted, size=(IMAGE_H, IMAGE_W))

    return image_resized

def plot_images(images, rows, index, original, cols=2):
    """
    print a row of plots
    Args:
        images:  image array
        rows:  number of rows of plots
        cols:  number of cols of plots
        index: index of plot
        original:  whether  picture is original
    """
    plot_title = ['original', 'mask']
    for i in range(len(images)):
        plt.axis('off')
        plt.title(plot_title[index])
        plt.subplot(rows, cols, index + 1)
        if original:
            plt.imshow(mpimg.imread(images[i]))
        else:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i]))
        index += 1


# split into arrays
folder = 'datasets/ISIC2018_Task1'
list_images = list(glob.glob(folder + '-2_Training_Input_x2/*.jpg'))
list_masks = list(glob.glob(folder + '_Training_GroundTruth_x2/*.png'))

# display original images
plt.figure(figsize=(4, 4))
if len(list_images) > NUM_EXAMPLES and len(list_masks) > NUM_EXAMPLES:
    for n in range(NUM_EXAMPLES):
        plot_images([list_images[n], list_masks[n]], NUM_EXAMPLES, n * 2, True)
    plt.show()
