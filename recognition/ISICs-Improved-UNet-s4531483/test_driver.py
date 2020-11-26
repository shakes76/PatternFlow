import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import layers_model as layers
import math
from itertools import islice
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # only print TF warnings and errors


# --------------------------------------------
# GLOBAL CONSTANTS
# --------------------------------------------

# Set the paths to data locations. The respective files for images/masks should be WITHIN A FOLDER WITHIN the below
# directories for the generators to work properly. For example, using the given ISICs data from
# https://cloudstor.aarnet.edu.au/sender/?s=download&token=505165ed-736e-4fc5-8183-755722949d34, folder structure would
# be \data\image\ISIC2018_Task1-2_Training_Input_x2 containing inputs, and
# \data\mask\ISIC2018_Task1_Training_GroundTruth_x2 containing ground truths.
PATH_ORIGINAL_DATA = os.path.join("data", "image")  # directory that contains folder containing input images
PATH_SEG_DATA = os.path.join("data", "mask")  # directory that contains folder containing ground truth images
IMAGE_HEIGHT = 512  # the height input images are scaled to
IMAGE_WIDTH = 512  # the width input images are scaled to
CHANNELS = 3  # the number of channels of the input image
SEED = 45  # set a seed so the different generators will work together properly, and match training/testing sets
BATCH_SIZE = 2  # set the batch_size
EPOCHS = 5  # set the number of epochs for training
LEARNING_RATE = 0.0005  # set the training learning rate
STEPS_PER_EPOCH_TRAIN = math.floor(2076 / BATCH_SIZE)  # set the number of steps per epoch for training samples
STEPS_PER_EPOCH_TEST = math.floor(519 / BATCH_SIZE)  # set the number of steps per epoch for testing samples
IMAGE_MODE = "rgb"  # image mode of input images (they are coloured rgb)
MASK_MODE = "grayscale"  # image mode of ground truth marks (they are binary black/white)
NUMBER_SHOW_TEST_PREDICTIONS = 3  # number of example test predictions to visualise after model has been trained
# Set the properties for the image generators for training images. Images transformed to help training generalisation.
DATA_TRAIN_GEN_ARGS = dict(
    rescale=1.0/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2)  # 0.2 used to have training set take the first 80% of images
# Set the properties for the image generators for testing images. No image transformations.
DATA_TEST_GEN_ARGS = dict(
    rescale=1.0/255,
    validation_split=0.8)  # 0.8 used to have test set take the final 20% of images (keep train/test data separated)
# Set the shared properties for generator flows - scale all images to given dimensions.
TEST_TRAIN_GEN_ARGS = dict(
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    interpolation="nearest",
    subset='training',  # all subsets are set to training - this corresponds to the first 80% and last 20% for each
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))


# --------------------------------------------
# MAIN FUNCTIONS
# --------------------------------------------

# Metric for how similar two sets (prediction vs truth) are.
# Implementation based off https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# DSC = (2|X & Y|) / (|X| + |Y|) -> 'soft' dice coefficient.
def dice_coefficient(truth, pred, eps=1e-7, axis=(1, 2, 3)):
    numerator = (2.0 * (tf.reduce_sum(pred * truth, axis=axis))) + eps
    denominator = tf.reduce_sum(pred, axis=axis) + tf.reduce_sum(truth, axis=axis) + eps
    dice = tf.reduce_mean(numerator / denominator)
    return dice


# Loss function - DSC distance.
def dice_loss(truth, pred):
    return 1.0 - dice_coefficient(truth, pred)


# Preprocess data forming the generators.
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

    # Ideally this would be a Sequence joining the two generators instead of zipping them together to keep everything
    # thread-safe, allowing for multiprocessing - but if it ain't broke. (It works).
    return zip(image_train_gen, mask_train_gen), zip(image_test_gen, mask_test_gen)


# Plot the accuracy and loss curves of model training.
def plot_accuracy_loss(track):
    plt.figure(0)
    plt.plot(track.history['accuracy'])
    plt.plot(track.history['loss'])
    plt.title('Loss & Accuracy Curves')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'])
    plt.show()


# Compile and train the model, evaluate test loss and accuracy.
def train_model_check_accuracy(train_gen, test_gen):
    model = layers.improved_unet(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss=dice_loss, metrics=['accuracy', dice_coefficient])
    track = model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        use_multiprocessing=False)
    plot_accuracy_loss(track)

    print("\nEvaluating test images...")
    test_loss, test_accuracy, test_dice = \
        model.evaluate(test_gen, steps=STEPS_PER_EPOCH_TEST, verbose=2, use_multiprocessing=False)
    print("Test Accuracy: " + str(test_accuracy))
    print("Test Loss: " + str(test_loss))
    print("Test DSC: " + str(test_dice) + "\n")
    return model


# Test and visualise model predictions with a set amount of test inputs.
def test_visualise_model_predictions(model, test_gen):
    test_range = np.arange(0, stop=NUMBER_SHOW_TEST_PREDICTIONS, step=1)
    figure, axes = plt.subplots(NUMBER_SHOW_TEST_PREDICTIONS, 3)
    for i in test_range:
        current = next(islice(test_gen, i, None))
        test_pred = model.predict(current, steps=1, use_multiprocessing=False)[0]
        truth = current[1][0]
        original = current[0][0]
        probabilities = keras.preprocessing.image.img_to_array(test_pred)
        test_dice = dice_coefficient(truth, test_pred, axis=None)

        axes[i][0].title.set_text('Input')
        axes[i][0].imshow(original, vmin=0.0, vmax=1.0)
        axes[i][0].set_axis_off()
        axes[i][1].title.set_text('Output (DSC: ' + str(test_dice.numpy()) + ")")
        axes[i][1].imshow(probabilities, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][1].set_axis_off()
        axes[i][2].title.set_text('Ground Truth')
        axes[i][2].imshow(truth, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][2].set_axis_off()
    plt.axis('off')
    plt.show()


# Run the test driver.
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
