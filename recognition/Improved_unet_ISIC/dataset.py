import tensorflow as tf
import os


def get_data(path, is_y=False):
    """

    :param is_y: to mark whether the output will be used as true image
    :param path: the parent folder of the folder contains all the images(contain '/' in the end)
    :return: a dataset with shape (n,128,128,1)

    """
    # to delete the superpixel images which are useless
    file = ''
    for f in os.listdir(path):
        if f == '.DS_Store':
            continue
        else:
            file = f
    inner_path = os.listdir(path + file + '/')
    for file_name in inner_path:
        if "superpixel" in file_name:
            os.remove(path + file + '/' + file_name)

    inner_path = os.listdir(path + file + '/')

    # use tf function ro get a resized to 64*64 tensor
    output = tf.keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode=None,
        color_mode='grayscale',
        image_size=(64, 64),
        batch_size=len(inner_path),
        shuffle=False
    )
    for element in output:
        output = element
        break


