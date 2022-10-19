import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def load_oasis():
    root_dir = r"C:\Users\jorda\Documents"
    directories = ["keras_png_slices_data\keras_png_slices_test",
                "keras_png_slices_data\keras_png_slices_train",
                "keras_png_slices_data\keras_png_slices_validate"]


    test_imgs, train_imgs = load_data2(root_dir, directories,4,32)

    train_np = np.stack(list(train_imgs))
    test_np = np.stack(list(test_imgs))
    # train_np = np.expand_dims(train_np, -1)
    # test_np = np.expand_dims(test_np, -1)
    # normalise
    # train_np = train_np/255
    # test_np = test_np/255

    var = np.var(train_np)
    # print(var)
    # # test_var = tf.
    # print(type(test_imgs))v

    print(type(train_np), train_np.shape)
    test_np = np.concatenate(test_np)
    train_np = np.concatenate(train_np)

    #np.reshape(test_np, (test_np.shape[0]*test_np.shape[1], 256, 256, 1))
    print(type(test_np), test_np.shape)

    return train_np, test_np, train_imgs, test_imgs, var

def load_data2(path, directories, batches1=32, batches2=32):
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

    # val_imgs = keras.utils.image_dataset_from_directory(os.path.join(path, directories[2]),
    #                                                     color_mode='grayscale',
    #                                                     label_mode = None,
    #                                                     batch_size=batches,
    #                                                     image_size=(256, 256))       
    # print(type(test_imgs))
    
    test_imgs = test_imgs.map(scale_down, num_parallel_calls=tf.data.AUTOTUNE)
    train_imgs = train_imgs.map(scale_down, num_parallel_calls=tf.data.AUTOTUNE)
    # val_imgs = val_imgs.map(scale_down, num_parallel_calls=tf.data.AUTOTUNE)
    # var = tfp.stats.variance(test_imgs)
    # print("variance is ", var)
    return test_imgs, train_imgs

def scale_down(image):
    return image/255