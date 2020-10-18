import os
import sys
from IPython.display import Image, display
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint

class CustomSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as TensorFlow arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: (i + self.batch_size)]
        batch_target_img_paths = self.target_img_paths[i: (i + self.batch_size)]
        x = tf.zeros([self.batch_size, self.img_size[0], self.img_size[1], 3], tf.float32)
        # x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = img_to_array(load_img(path, target_size=self.img_size))
            print(tf.shape(img))
            print(tf.shape(x))
            x[j, :, :, :] = img
        y = tf.zeros([self.batch_size, self.img_size[0], self.img_size[1], 3], tf.uint8)
        # y = np.zeros((batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = img_to_array(load_img(path, target_size=self.img_size, color_mode="grayscale"))
            y[j, :, :] = tf.expand_dims(img, 2)
        return x, y

def get_img_target_paths(img_dir, seg_dir):

    input_img_paths = sorted(
        [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(seg_dir, fname)
            for fname in os.listdir(seg_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    # print("Number of cases samples:", len(input_img_paths))
    # print("Number of segmentation samples:", len(target_img_paths))
    # for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    #     print(input_path, "|", target_path)

    return input_img_paths, target_img_paths


def get_img_sizes(input_paths):
    sizes = [Image.open(f, 'r').size for f in input_paths]
    return max(sizes), min(sizes)


def get_model(img_size, num_classes):
    # Free up RAM in case the model definition cells were run multiple times
    # keras.backend.clear_session()

    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def split(val_split, input_img_paths, target_img_paths, test_split=0.05, seed=1337):
    # Split our img paths into a training and a validation set
    test_samples = int(test_split * len(input_img_paths))
    val_samples = int(val_split * (len(input_img_paths) - test_samples))
    train_samples = len(input_img_paths) - test_samples - val_samples
    print("\nTrain: ", str(len(input_img_paths) - test_samples - val_samples),"\nTest: ",  str(test_samples), "\nVal: ", str(val_samples))

    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:train_samples]
    train_target_img_paths = target_img_paths[:train_samples]
    val_input_img_paths = input_img_paths[train_samples:(train_samples+val_samples)]
    val_target_img_paths = target_img_paths[train_samples:(train_samples+val_samples)]
    test_input_img_paths = input_img_paths[(train_samples+val_samples):]
    test_target_img_paths = target_img_paths[(train_samples+val_samples):]

    return train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths, test_input_img_paths, test_target_img_paths

def create_generator(img_size, batch_size, img_paths, target_paths):

    # Instantiate data Sequences for each split
    gen = CustomSequence(batch_size, img_size, img_paths, target_paths)
    # val_gen = CustomSequence(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    # test_gen = CustomSequence(batch_size, img_size, test_input_img_paths, test_target_img_paths)

    return gen


def train(train_gen, val_gen, model, epochs=15, save_model_filename=None, save_path=None):

    if save_model_filename:
        checkpoint = ModelCheckpoint(filepath=save_path,
                                     save_weights_only=True,
                                     #                              save_best_model,
                                     verbose=1)
        callbacks = [checkpoint]

        # callbacks = [
        #     keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
        # ]

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    # Train the model, doing validation at the end of each epoch.

    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    ## Save model:
    if save_path:
        print("model_saved\n")
        model.save(save_path)

    return model


def evaluate(test_gen, model):

    return model.predict(test_gen)


def display_mask(test_pred):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(test_pred, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)


def local_display(i, val_input_img_paths, val_target_img_paths):
    # Display input image
    display(Image(filename=val_input_img_paths[i]))

    # Display ground-truth target mask
    img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
    display(img)

    # Display mask predicted by our model
    display_mask(i)  # Note that the model only sees inputs at 150x150.


if __name__ == "__main__":

    # black and white segmentation
    num_classes = 2
    # how many images to run through net:
    batch_size = 2

    img_dir = "H:\COMP3710\ISIC2018_Task1-2_Training_Input_x2"
    seg_dir = "H:\COMP3710\ISIC2018_Task1_Training_GroundTruth_x2"
    save_model_filename = ".\\model_test"
    save_path = ".\\training_test\\cp.ckpt"

    input_img_paths, target_img_paths = get_img_target_paths(img_dir, seg_dir)

    max_img_size, min_img_size = get_img_sizes(input_img_paths)

    print(max_img_size)
    print(min_img_size)

    # model = get_model(img_size, num_classes)
    #
    # train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths, test_input_img_paths, test_target_img_paths = split(0.2, input_img_paths, target_img_paths)
    # print("array lengths:\nTrain: ", str(len(train_input_img_paths)), "\nTest: ", str(len(test_input_img_paths)),
    #       "\nVal: ", str(len(val_input_img_paths)))
    #
    # train_gen = create_generator(img_size, batch_size, train_input_img_paths, train_target_img_paths)
    # val_gen = create_generator(img_size, batch_size, val_input_img_paths, val_target_img_paths)
    # test_gen = create_generator(img_size, batch_size, test_input_img_paths, test_target_img_paths)
    #
    # # train_gen, val_gen, test_gen = get_generators(0.8, img_size, input_img_paths, target_img_paths)
    #
    # model = train(train_gen, val_gen, model, epochs=15, save_model_filename=save_model_filename, save_path=save_path)
    #
    # test_preds = evaluate(test_gen, model)
    #
    # print_array = list(range(2))
    # if len(print_array) > 0:
    #     for i in range(len(print_array)):
    #         local_display(i, test_input_img_paths, test_target_img_paths)
    #         display(test_preds[i])
