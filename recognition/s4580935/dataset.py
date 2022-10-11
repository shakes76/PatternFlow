import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import datasets, layers, models, backend
#import matplotlib.pyplot as plt
#from PIL import Image
import numpy as np
import IPython.display as display
import pathlib
import glob

def FetchData(path):
    images = sorted(glob.glob(path))
    return images

def ImageExtract(images):
    Array = []
    for x in images:
        sample_image = tf.io.read_file(str(x))
        sample_image = tf.image.decode_png(sample_image)
        sample_image = tf.image.convert_image_dtype(sample_image, tf.float32)
        sample_image = tf.image.resize(sample_image, [256, 256])
        y = tf.shape(sample_image)[1]
        y = y // 2
        image = sample_image[:, y:, :]
        image = tf.cast(sample_image, tf.float32)
        image = np.squeeze(image)
        Array.append([image])
    return Array



def main():
    train_images = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_train\\*')
    train_labels = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_seg_train\\*')

    test_images = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_test\\*')
    test_labels = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_seg_test\\*')

    validate_images = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_validate\\*')
    validate_labels = FetchData('D:\\University\\2022 Sem 2\\COMP3710\\Project Report\\BranchFOlder\\keras_png_slices_data\\keras_png_slices_data\\keras_png_slices_seg_validate\\*')

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validate_ds = tf.data.Dataset.from_tensor_slices((validate_images, validate_labels))
    print(len(train_images))
    print(len(test_images))
    print(len(validate_images))

    train_ds = train_ds.shuffle(len(train_images))
    val_ds = validate_ds.shuffle(len(validate_images))
    trainArray = ImageExtract(train_images)
    testArray = ImageExtract(test_images)
    validateArray = ImageExtract(validate_images)
    train = np.asarray(trainArray)
    train = np.squeeze(train)

    test = np.asarray(testArray)
    test = np.squeeze(test)

    validate = np.asarray(validateArray)
    validate = np.squeeze(validate)

if __name__ =="__main__":
    main();
