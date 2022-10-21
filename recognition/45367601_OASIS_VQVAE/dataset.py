import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def load_oasis(root_dir = r"C:\Users\jorda\Documents"):
    root_dir = r"C:\Users\jorda\Documents"
    directories = ["keras_png_slices_data\keras_png_slices_test",
                "keras_png_slices_data\keras_png_slices_train",
                "keras_png_slices_data\keras_png_slices_validate"]


    test_imgs, train_imgs = load_data2(root_dir, directories,4,32)
    # make numpy arrays of data.
    train_np = np.stack(list(train_imgs))
    test_np = np.stack(list(test_imgs))
    # the variance of train set
    var = np.var(train_np)
    test_np = np.concatenate(test_np)
    train_np = np.concatenate(train_np)

    return train_np, test_np, train_imgs, test_imgs, var

def load_data2(path, directories, batches1=32, batches2=32):
    # Load the dataset
    test_imgs = keras.utils.image_dataset_from_directory(os.path.join(path, directories[0]),
                                                        color_mode='grayscale',
                                                        label_mode = None,
                                                        batch_size=batches2,
                                                        image_size=(256, 256))       

    train_imgs = keras.utils.image_dataset_from_directory(os.path.join(path, directories[1]),
                                                        color_mode='grayscale',
                                                        label_mode = None,
                                                        batch_size=batches1,
                                                        image_size=(256, 256))       


    
    test_imgs = test_imgs.map(scale_down, num_parallel_calls=tf.data.AUTOTUNE)
    train_imgs = train_imgs.map(scale_down, num_parallel_calls=tf.data.AUTOTUNE)

    return test_imgs, train_imgs

def scale_down(image):
    return image/255