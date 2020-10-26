import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import layers_model as layers
import math
from itertools import islice

PATH_ORIGINAL_DATA = "data/image"
PATH_SEG_DATA = "data/mask"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
CHANNELS = 3
SEED = 45
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 0.0004
STEPS_PER_EPOCH_TRAIN = math.floor(2076 / BATCH_SIZE)
STEPS_PER_EPOCH_TEST = math.floor(519 / BATCH_SIZE)
IMAGE_MODE = "rgb"
MASK_MODE = "grayscale"
NUMBER_SHOW_TEST_PREDICTIONS = 3
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
    subset='training',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))


def pre_process_data():
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TRAIN_GEN_ARGS)
    train_mask_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TRAIN_GEN_ARGS)
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TEST_GEN_ARGS)
    test_mask_data_generator = keras.preprocessing.image.ImageDataGenerator(**DATA_TEST_GEN_ARGS)

    image_train_gen = train_image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        color_mode=IMAGE_MODE,
        **TEST_TRAIN_GEN_ARGS)

    image_test_gen = test_image_data_generator.flow_from_directory(
        PATH_ORIGINAL_DATA,
        color_mode=IMAGE_MODE,
        **TEST_TRAIN_GEN_ARGS)

    mask_train_gen = train_mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        color_mode=MASK_MODE,
        **TEST_TRAIN_GEN_ARGS)

    mask_test_gen = test_mask_data_generator.flow_from_directory(
        PATH_SEG_DATA,
        color_mode=MASK_MODE,
        **TEST_TRAIN_GEN_ARGS)

    return zip(image_train_gen, mask_train_gen), zip(image_test_gen, mask_test_gen)


def plot_accuracy_loss(track):
    plt.figure(0)
    plt.plot(track.history['accuracy'])
    plt.plot(track.history['loss'])
    plt.title('Loss & Accuracy Curves')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'])
    plt.show()


def train_model_check_accuracy(train_gen, test_gen):
    model = layers.improved_unet(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    track = model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        use_multiprocessing = False)
    # plot_accuracy_loss(track)

    print("\nEvaluating test images...")
    test_loss, test_accuracy = model.evaluate(test_gen, steps=STEPS_PER_EPOCH_TEST, verbose=2, use_multiprocessing=False)
    print("Test Accuracy: " +  str(test_accuracy))
    print("Test Loss: " + str(test_loss) + "\n")
    return model


def test_visualise_model_predictions(model, test_gen):
    test_range = np.arange(0, stop=NUMBER_SHOW_TEST_PREDICTIONS, step=1)
    test_gen_subset = np.zeros(test_range.shape)
    for i in test_range:
        current = next(islice(test_gen, i, None))
        test_pred = model.predict(current, steps=1, use_multiprocessing=False)[0]
        truth = current[1][0]
        original = current[0][0]
        probabilities = keras.preprocessing.image.img_to_array(test_pred)
        ones = probabilities >= 0.5
        zeroes = probabilities < 0.5
        thresholded = probabilities
        thresholded[ones] = 1
        thresholded[zeroes] = 0
        figure, axes = plt.subplots(1, 4)
        axes[0].title.set_text('Output')
        axes[0].imshow(probabilities, cmap='gray', vmin=0.0, vmax=1.0)
        axes[1].title.set_text('Thresholded')
        axes[1].imshow(thresholded, cmap='gray', vmin=0.0, vmax=1.0)
        axes[2].title.set_text('Input')
        axes[2].imshow(original, vmin=0.0, vmax=1.0)
        axes[3].title.set_text('Model Output')
        axes[3].imshow(truth, cmap='gray',vmin=0.0, vmax=1.0)
        plt.show()


def main():
    print("\nPREPROCESSING IMAGES")
    train_gen, test_gen = pre_process_data()
    print("\nTRAINING MODEL")
    model = train_model_check_accuracy(train_gen, test_gen)
    print("\nVISUALISING PREDICTIONS")
    test_visualise_model_predictions(model, test_gen)

    print("COMPLETED")
    return 0


if __name__ == "__main__":
    main()
