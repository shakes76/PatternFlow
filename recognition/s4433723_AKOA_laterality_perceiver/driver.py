from model import *
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Global variables for directory setting
DATASET_DIR = 'datasets/'
OUTPUT_DIR = 'output/'
INPUT_DS_SIZE = 18680

# Hyper-parameters
IMG_SIZE = 16  # size to resize input images
NUM_CLASSES = 2
NUM_CHANNELS = 1
LEARN_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 1
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
PATCH_SIZE = 2  # Size of patches to be extracted from input images.
PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # Size of the img data array.
LATENT_ARRAY_SIZE = 128  # Size of the latent array.
PROJECTION_SIZE = 128  # Embedding size of each element in the data and latent arrays.
NUM_HEADS = 8  # Number of transformer heads.
dense_units = [PROJECTION_SIZE, PROJECTION_SIZE] # Size of the Transformer Dense network.
num_transformer_blocks = 4
num_iterations = 2  # Repetitions of the cross-attention and Transformer modules.

# Size of the Dense network of the final classifier.
classifier_units = [PROJECTION_SIZE, NUM_CLASSES]

print(f"Image size: {IMG_SIZE} X {IMG_SIZE} = {IMG_SIZE ** 2}")
print(f"Patch size: {PATCH_SIZE} X {PATCH_SIZE} = {PATCH_SIZE ** 2} ")
print(f"Patches per image: {PATCHES}")
print(f"Elements per patch: {(PATCH_SIZE ** 2) * NUM_CHANNELS}")
print(f"Latent array shape: {LATENT_ARRAY_SIZE} X {PROJECTION_SIZE}")
print(f"Data array shape: {PATCHES} X {PROJECTION_SIZE}")


def load_train_test_data(dataset_dir):
    """

    :param dataset_dir:
    :return:
    """

    # count total items in dataset dir
    #input_ds_size = 0
    #print(input_ds_size)
    input_ds_size = 0
    for _, _, files in os.walk(dataset_dir):
        input_ds_size += len(files)
    print(input_ds_size)
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

    # save e.g. image from ds
    first_image = x_data[0, :, :, 0]
    plt.imsave(OUTPUT_DIR + "input_train_image.png", first_image, format="png", cmap=plt.cm.gray)

    # train test split
    return train_test_split(x_data, y_data, test_size=0.25, random_state=42)


if __name__ == "__main__":

    # check tf version
    print("Tensorflow version: ", tf.__version__)

    # if need to sort akoa dataset into datasets folder, do so:

    # load in data from processed directory
    x_train, x_test, y_train, y_test = load_train_test_data(DATASET_DIR)
    print("Input Shapes:")
    print("X train: ", x_train.shape)
    print("Y train: ", y_train.shape)
    print("X test: ", x_test.shape)
    print("Y test: ", y_test.shape)

    # Create perceiver model
    perceiver = Perceiver(
        PATCH_SIZE,
        PATCHES,
        LATENT_ARRAY_SIZE,
        PROJECTION_SIZE,
        NUM_HEADS,
        num_transformer_blocks,
        dense_units,
        DROPOUT_RATE,
        num_iterations,
        classifier_units,
    )

    # Create LAMB optimizer with weight decay.
    lamb = tfa.optimizers.LAMB(
        learning_rate=LEARN_RATE, weight_decay_rate=WEIGHT_DECAY,
    )

    # Compile the model.
    perceiver.compile(
        optimizer=lamb,#tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    # fit model with 80:20 train/validation split
    history = perceiver.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
    )

    # visualise shape of model
    perceiver.summary()

    test_loss, test_accuracy = perceiver.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss * 100:.2f}%")

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{OUTPUT_DIR}training_curve_epochs_{EPOCHS}.png')