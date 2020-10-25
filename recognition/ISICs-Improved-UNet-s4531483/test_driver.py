import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import layers_model as layers
import math

PATH_ORIGINAL_DATA = "data/image"
PATH_SEG_DATA = "data/mask"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
SEED = 45
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 0.0005
STEPS_PER_EPOCH_TRAIN = math.ceil(2076 / BATCH_SIZE)
STEPS_PER_EPOCH_TEST = math.ceil(518 / BATCH_SIZE)
DATA_GEN_ARGS = dict(
    rescale=1.0/255,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2)
TEST_TRAIN_GEN_ARGS = dict(
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    interpolation="nearest",
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))


if __name__ == "__main__":
    image_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_GEN_ARGS)
    mask_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_GEN_ARGS)

    image_train_gen = image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='training',
        color_mode='rgb')

    image_test_gen = image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='validation',
        color_mode='rgb')

    mask_train_gen = mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='training',
        color_mode='grayscale')

    mask_test_gen = mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='validation',
        color_mode='grayscale')

    model = layers.improvedUNet(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    train_gen = zip(image_train_gen, mask_train_gen)
    test_gen = zip(image_test_gen, mask_test_gen)

    track = model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
    )

    print("\nTest Accuracy:")
    test_loss, test_accuracy = model.evaluate(test_gen, steps=STEPS_PER_EPOCH_TEST, verbose=2)

    plt.plot(track.history['accuracy'])
    plt.plot(track.history['loss'])
    plt.title('Loss & Accuracy Curves')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'])
    plt.show()

    print("COMPLETED.")
