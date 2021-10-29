# Start by importing dependencies.

import pathlib
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import PIL.Image
import glob
import matplotlib.pyplot as plt
import keras.backend as K

# Import the model.
from unet_isic import get_UNET_model
from unet_isic import get_model

# Setup the datafiles directories.
image_dir = './data/images/'
mask_dir = './data/masks/'
img_height, img_width = 128, 128


# Some stuff to bug fix.
def verify_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # os.getcwd()


def decode_image(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [img_height, img_width])


def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = decode_image(img)
    mask = tf.io.read_file(mask_path)
    mask = decode_mask(mask)
    return img, mask


def normalize_dataset(image, mask):
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    image = normalization_layer(image)
    mask = normalization_layer(mask)
    return image, mask


def batch_normalized_dataset(normalized_ds, dataset_size):
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000

    TRAIN_DATASET_SIZE = int(dataset_size * 0.6)
    VALIDATION_DATASET_SIZE = int(dataset_size * 0.2)
    TEST_DATASET_SIZE = int(dataset_size * 0.2)
    normalized_ds = (
        normalized_ds
        .shuffle(BUFFER_SIZE)
    )

    train_batches = (
        normalized_ds
        .take(TRAIN_DATASET_SIZE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_batches = (
        normalized_ds
        .skip(TRAIN_DATASET_SIZE)
        .take(VALIDATION_DATASET_SIZE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_batches = (
        normalized_ds
        .skip(TRAIN_DATASET_SIZE + VALIDATION_DATASET_SIZE)
        .take(TEST_DATASET_SIZE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_batches, validation_batches, test_batches


def create_mask(pred_mask):
    pred_mask = tf.math.round(pred_mask)
    # pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(model, dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            print("Predicted mask shape: ", pred_mask[0].shape)
            print("Max pixel value in predicted mask: ", np.max(pred_mask))
            display([image[0], mask[0], create_mask(pred_mask[0])])
    # else:
    #   display([sample_image, sample_mask,
    #            create_mask(model.predict(sample_image[tf.newaxis, ...]))])


def dice_coeff(y_true, y_pred, axis=(1, 2, 3)):
    # print("y shape: ", y_true.shape)
    # y_true = tf.math.round(y_pred)
    num_y_true = K.sum(y_true, axis=axis)

    y_pred_mask = tf.math.round(y_pred)
    num_y_pred = K.sum(y_pred_mask, axis=axis)

    intersection = tf.math.multiply(y_true, y_pred_mask)

    num_intersection = K.sum(intersection, axis=axis)

    smoothing = 0.0000001
    return (2 * num_intersection + smoothing) / (num_y_true + num_y_pred + smoothing)




def main():
    print(tf.__version__)
    verify_gpu()
    tf.compat.v1.enable_eager_execution()
    root_image_dir = pathlib.Path(image_dir)
    root_mask_dir = pathlib.Path(mask_dir)

    # See the first image in the directory
    PIL.Image.open(list(root_image_dir.glob('*.jpg'))[0])
    PIL.Image.open(list(root_mask_dir.glob('*.png'))[0])

    # Count the number of images in both directories.
    image_files = sorted(glob.glob(image_dir + '*.jpg'))
    mask_files = sorted(glob.glob(mask_dir + '*.png'))
    image_count = len(image_files)
    mask_count = len(mask_files)
    dataset_size = image_count
    print('Number of images: ', image_count)
    print('Number of masks: ', mask_count)

    train_ds = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    train_ds = train_ds.map(process_path)
    for element in train_ds.take(1):
        print("Image shape: ", element[0].numpy().shape, "\n", "Mask shape: ", element[1].numpy().shape)
        # print("First image tensor: ", element[0].numpy())
        # print("First mask tensor: ", element[1].numpy())

    normalized_ds = train_ds.map(normalize_dataset)
    for element in normalized_ds.take(1):
        print("Image shape: ", element[0].numpy().shape, "\n", "Mask shape: ", element[1].numpy().shape)
        # print("First image tensor: ", element[0].numpy())
        # print("First mask tensor: ", element[1].numpy())
        print("Max pixel value in mask: ", np.max(element[1].numpy()))

    # Batch the dataset
    train_batches, validation_batches, test_batches = batch_normalized_dataset(normalized_ds, dataset_size)

    model = get_UNET_model()
    # meanIoU = tf.keras.metrics.MeanIoU(num_classes=2)
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy", dice_coeff])

    print(model.summary())

    epochs = 50
    model.fit(train_batches, epochs=epochs, validation_data=validation_batches)

    model.evaluate(test_batches)
    show_predictions(model=model, dataset=train_batches)


if __name__ == "__main__":
    main()
