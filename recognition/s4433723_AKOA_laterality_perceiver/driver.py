"""
This file is the main script that processes input data and trains the perceiver model,
producing associated output data.
"""
from model import *
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

# Global variables for directory and input data settings
ORIGINAL_IMG_DIR = "AKOA_Analysis/"
CLASSES = ("Left", "Right")
NUM_CLASSES = len(CLASSES)
DATASET_DIR = 'datasets/'

# Training hyper-parameters
IMG_SIZE = 16  # size to resize input images
NUM_CHANNELS = 1  # grayscale images have one channel
EPOCHS = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.25
LR = 0.001
WD = 0.0001


def underscore_modify(string):
    """
    Quick helper to allow for appropriate sorting of class labels in AKOA dataset.
    :param string: string to modify with underscores
    :return: the modified string
    """

    modified_string = ""
    for char in string:
        modified_string += f"_{char}"
    return modified_string


def create_sorted_data_directory(img_dir, classes):
    """
    Take images from input AKOA dataset folder and place them in datasets
    folder in sub-directories according to their class heading. This is
    to help in loading data by class later.
    :param img_dir: the directory to load data from
    :param classes: the classes to sort into sub-folders from AKOA dataset.
    :return:
    """

    # create subdirectories in dataset folder
    for class_name in classes:
        try:
            os.mkdir(f"{DATASET_DIR}{class_name}")
        except OSError as e:
            print(e)

    # move files from input dataset into datasets folder, in class sub-dirs
    for file_name in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file_name)
        for class_name in classes:
            if class_name in file_name \
                    or class_name.upper() in file_name \
                    or underscore_modify(class_name.upper()) in file_name \
                    or class_name.lower() in file_name:
                os.rename(file_path, f"{DATASET_DIR}{class_name}/{file_name}")

def load_train_test_data(dataset_dir, test_split):
    """
    Loads images from datasets folder with class subdirectories, shuffles the data, resizes it
    to the given IMG_SIZE by bilinear interpolation, normalises the grayscale images and then
    returns the data as a train test split of numpy arrays.

    :param dataset_dir: the directory to extract training data from
    :return: a train test split of numpy arrays
    """

    # count total items in dataset dir
    input_ds_size = 0
    for _, _, files in os.walk(dataset_dir):
        input_ds_size += len(files)

    # load AKOA dataset from processed datasets directory
    akoa_ds_tuple = tf.keras.preprocessing.image_dataset_from_directory(directory=dataset_dir,
                                                                        shuffle=True,
                                                                        seed=999,
                                                                        image_size=(IMG_SIZE, IMG_SIZE),
                                                                        batch_size=input_ds_size,
                                                                        labels="inferred",
                                                                        label_mode="categorical",
                                                                        color_mode="grayscale",
                                                                        ),

    # extract dataset from tuple
    akoa_ds = akoa_ds_tuple[0]

    # normalise input images
    normaliser = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1),
    ])
    akoa_ds = akoa_ds.map(lambda x, y: (normaliser(x), y))

    # get data into numpy arrays
    x_data, y_data = next(iter(akoa_ds))
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # plot grid of images
    f, axs = plt.subplots(8, 8)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow(x_data[i, :, :, 0], cmap=plt.cm.gray)
        ax.axis('off')
    plt.tight_layout()
    plt.title("Normalised input images")
    plt.show()

    # train test split
    return train_test_split(x_data, y_data, test_size=0.25, random_state=42)


if __name__ == "__main__":

    # Handle cmd line args just for input image sorting
    argv = sys.argv
    argc = len(argv)
    if argc == 2:
        input_img_dir = argv[1]
        print(f"Loading in data from {input_img_dir}...")
    else:
        input_img_dir = ORIGINAL_IMG_DIR
        print("Defaulting to AKOA_Analysis input directory for sorting...")

    # check tf version
    print("Tensorflow version: ", tf.__version__)

    # if need to sort akoa dataset into datasets folder, do so:
    if "datasets" not in os.listdir(".") \
            or "LEFT" not in os.listdir(DATASET_DIR) \
            or "RIGHT" not in os.listdir(DATASET_DIR):
        os.mkdir("datasets")
        print(f"Sorting {ORIGINAL_IMG_DIR} into datasets folder...")
        create_sorted_data_directory(ORIGINAL_IMG_DIR, CLASSES)
    else:
        print("Datasets folder already exists...")

    # load in data from processed directory
    x_train, x_test, y_train, y_test = load_train_test_data(DATASET_DIR, TEST_SPLIT)
    print("Input Shapes:")
    print("X train: ", x_train.shape)
    print("Y train: ", y_train.shape)
    print("X test: ", x_test.shape)
    print("Y test: ", y_test.shape)

    # build perceiver, pass in keyword args here to modify defaults
    perceiver = Perceiver(IMG_SIZE, NUM_CLASSES, latent_array_size=32)

    # compile perceiver
    perceiver.compile(
        optimizer=tfa.optimizers.LAMB(learning_rate=LR, weight_decay_rate=WD),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    # fit model with 80:20 train/validation split
    history = perceiver.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )

    # visualise shape of model
    perceiver.summary()

    # evaluate model performance on test data
    test_loss, test_accuracy = perceiver.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss * 100:.2f}%")

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Perceiver Model Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'training_curve_epochs_{EPOCHS}.png')