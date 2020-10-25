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
EPOCHS = 1
LEARNING_RATE = 0.0004
STEPS_PER_EPOCH_TRAIN = math.floor(2076 / BATCH_SIZE)
STEPS_PER_EPOCH_TEST = math.floor(519 / BATCH_SIZE)
DATA_TRAIN_GEN_ARGS = dict(
    rescale=1.0/255,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2)
DATA_TEST_GEN_ARGS = dict(
    rescale=1.0/255,
    validation_split=0.8)
TEST_TRAIN_GEN_ARGS = dict(
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    interpolation="nearest",
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))


if __name__ == "__main__":
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TRAIN_GEN_ARGS)
    train_mask_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TRAIN_GEN_ARGS)
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TEST_GEN_ARGS)
    test_mask_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TEST_GEN_ARGS)

    image_train_gen = train_image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='training',
        color_mode='rgb')

    image_test_gen = test_image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='training',
        color_mode='rgb')

    mask_train_gen = train_mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='training',
        color_mode='grayscale')

    mask_test_gen = test_mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        **TEST_TRAIN_GEN_ARGS,
        subset='training',
        color_mode='grayscale')

    model = layers.improvedUNet(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
    model.summary()
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

    test_pred = model.predict(test_gen, steps=STEPS_PER_EPOCH_TEST)

    mask = keras.preprocessing.image.img_to_array(test_pred[10])
    ones = mask >= 0.5
    zeroes = mask < 0.5
    mask[ones] = 1
    mask[zeroes] = 0

    plt.figure(0)
    plt.imshow(mask, cmap='gray', vmin=0.0, vmax=1.0)

    plt.figure(1)
    plt.plot(track.history['accuracy'])
    plt.plot(track.history['loss'])
    plt.title('Loss & Accuracy Curves')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'])
    plt.show()

    print("COMPLETED.")
