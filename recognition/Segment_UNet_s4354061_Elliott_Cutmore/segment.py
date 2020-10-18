import os
import sys
from IPython.display import Image, display
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Concatenate, UpSampling2D, Conv2DTranspose, Dense
from tensorflow.keras import Model
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from operator import itemgetter

class CustomSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as TensorFlow arrays)."""

    def __init__(self, batch_size, img_dims, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: (i + self.batch_size)]
        batch_target_img_paths = self.target_img_paths[i: (i + self.batch_size)]


        for j, path in enumerate(batch_input_img_paths):
            img = img_to_array(load_img(path))
            img = img / 255.
            img = tf.image.resize_with_crop_or_pad(img, self.img_dims[0], self.img_dims[1])
            img = tf.cast(img, tf.float32)
            img = tf.reshape(img, [self.img_dims[0] * self.img_dims[1] * self.img_dims[2]])
            if j == 0:
                x = img
            else:
                x = tf.concat([x, img], axis=0)

        for j, path in enumerate(batch_target_img_paths):
            img = img_to_array(load_img(path, color_mode="grayscale"))
            img = img / 255
            img = tf.image.resize_with_crop_or_pad(img, self.img_dims[0], self.img_dims[1])
            tf.reshape(img, [self.img_dims[0] * self.img_dims[1]])
            if j == 0:
                y = img
            else:
                y = tf.concat([y, img], axis=0)

        x = tf.reshape(x, [batch_size, self.img_dims[0], self.img_dims[1], self.img_dims[2]])
        y = tf.reshape(y, [batch_size, self.img_dims[0], self.img_dims[1], 1])

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
    # max_w = max(sizes, key=lambda item: item[0])
    # max_h = max(sizes, key=lambda item: item[1])
    # min_w = min(sizes, key=lambda item: item[0])
    # min_h = min(sizes, key=lambda item: item[1])
    # return max_w, max_h, min_w, min_h
    return sizes


def get_model(img_dims, num_classes):
    # Free up RAM in case the model definition cells were run multiple times
    # keras.backend.clear_session()

    act = 'relu'
    kern = 'he_uniform'
    pad = 'same'
    inter = 'nearest'
    f = [64, 128, 256, 512, 1024]

    # Input layer
    # input_layer = layers.Input(shape=train[0].shape)
    input_layer = Input(shape=img_dims, name="Input")

    ## Convolutional layers - Feature learning
    # VGG 1:
    conv_1_1 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_1_1")(input_layer)
    conv_1_2 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_1_2")(conv_1_1)
    pool_1 = MaxPooling2D((2, 2), strides=(2, 2), name="pool_1")(conv_1_2)

    # VGG 2:
    conv_2_1 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_2_1")(pool_1)
    conv_2_2 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_2_2")(conv_2_1)
    pool_2 = MaxPooling2D((2, 2), strides=(2, 2), name="pool_2")(conv_2_2)

    # VGG 3:
    conv_3_1 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_3_1")(pool_2)
    conv_3_2 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_3_2")(conv_3_1)
    pool_3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool_3")(conv_3_2)

    # VGG 4:
    conv_4_1 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_4_1")(pool_3)
    conv_4_2 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_4_2")(conv_4_1)
    pool_4 = MaxPooling2D((2, 2), strides=(2, 2), name="pool_4")(conv_4_2)

    # Bottom VGG:
    bot_1 = Conv2D(f[4], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="bot_1")(pool_4)
    bot_2 = Conv2D(f[4], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="bot_2")(bot_1)

    # CONCAT 4:
    # cat_4_1 = Conv2DTranspose(f[3], (2, 2), strides=(2, 2), name="up_4")(bot_2)
    cat_4_1 = UpSampling2D((2, 2), interpolation=inter, name="up_4")(bot_2)
    cat_4_1 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_x")(cat_4_1)
    cat_4_2 = Concatenate(axis=3, name="cat_4")([conv_4_2, cat_4_1])
    cat_4_3 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_1")(cat_4_2)
    cat_4_4 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_2")(cat_4_3)

    # CONCAT 3:
    # cat_3_1 = Conv2DTranspose(f[2], (2, 2), strides=(2, 2), name="up_3")(cat_4_4)
    cat_3_1 = UpSampling2D((2, 2), interpolation=inter, name="up_3")(cat_4_4)
    cat_3_1 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_x")(cat_3_1)
    cat_3_2 = Concatenate(axis=3, name="cat_3")([conv_3_2, cat_3_1])
    cat_3_3 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_1")(cat_3_2)
    cat_3_4 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_2")(cat_3_3)

    # CONCAT 2:
    # cat_2_1 = Conv2DTranspose(f[1], (2, 2), strides=(2, 2), name="up_2")(cat_3_4)
    cat_2_1 = UpSampling2D((2, 2), interpolation=inter, name="up_2")(cat_3_4)
    cat_2_1 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_x")(cat_2_1)
    cat_2_2 = Concatenate(axis=3, name="cat_2")([conv_2_2, cat_2_1])
    cat_2_3 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_1")(cat_2_2)
    cat_2_4 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_2")(cat_2_3)

    # CONCAT 1:
    # cat_1_1 = Conv2DTranspose(f[0], (2, 2), strides=(2, 2), name="up_1")(cat_2_4)
    cat_1_1 = UpSampling2D((2, 2), interpolation=inter, name="up_1")(cat_2_4)
    cat_1_1 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_x")(cat_1_1)
    cat_1_2 = Concatenate(axis=3, name="cat_1")([conv_1_2, cat_1_1])
    cat_1_3 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_1")(cat_1_2)
    cat_1_4 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_2")(cat_1_3)

    # Output layer
    output_layer = Conv2D(num_classes, (1, 1), activation="softmax", kernel_initializer=kern, padding=pad,
                          name="Output")(cat_1_4)
    # output_layer = cat_1_4

    # Create model:
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def train_test_val_split(val_split, input_img_paths, target_img_paths, test_split=0.05, seed=1337):
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

def create_generator(img_dims, batch_size, img_paths, target_paths):

    # Instantiate data Sequences for each split
    gen = CustomSequence(batch_size, img_dims, img_paths, target_paths)
    # val_gen = CustomSequence(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    # test_gen = CustomSequence(batch_size, img_size, test_input_img_paths, test_target_img_paths)

    return gen


def train_model(train_gen, val_gen, model, epochs=15, save_model_filename=None, save_path=None):

    print("Some BS")

    if save_model_filename:
        checkpoint = ModelCheckpoint(filepath=save_path,
                                     save_weights_only=True,
                                     #                              save_best_model,
                                     verbose=1)
        callbacks = [checkpoint]
    else:
        callbacks = []

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
    batch_size = 4

    img_dir = "H:\COMP3710\ISIC2018_Task1-2_Training_Input_x2"
    seg_dir = "H:\COMP3710\ISIC2018_Task1_Training_GroundTruth_x2"
    save_model_filename = ".\\model_test"
    save_path = ".\\training_test\\cp.ckpt"

    input_img_paths, target_img_paths = get_img_target_paths(img_dir, seg_dir)

    # max_w, max_h, min_w, min_h = get_img_sizes(input_img_paths)
    # sizes = get_img_sizes(input_img_paths)
    # max_w = max(sizes, key=lambda item: item[0])
    # max_h = max(sizes, key=lambda item: item[1])
    # min_w = min(sizes, key=lambda item: item[0])
    # min_h = min(sizes, key=lambda item: item[1])

    img_dims = (384, 512, 3)
    img_size = (384, 512)

    # print("max_h: ", str(max_h), "\nmax_w: ", str(max_w), "\nmin_h: ", str(min_h), "\nmin_w: ", str(min_w))

    model = get_model(img_size, num_classes)

    train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths, test_input_img_paths, test_target_img_paths = split(0.2, input_img_paths, target_img_paths)
    print("array lengths:\nTrain: ", str(len(train_input_img_paths)), "\nTest: ", str(len(test_input_img_paths)),
          "\nVal: ", str(len(val_input_img_paths)))

    train_gen = create_generator(img_dims, batch_size, train_input_img_paths, train_target_img_paths)
    val_gen = create_generator(img_dims, batch_size, val_input_img_paths, val_target_img_paths)
    test_gen = create_generator(img_dims, batch_size, test_input_img_paths, test_target_img_paths)
    #
    model = train_model(train_gen, val_gen, model, epochs=15 ) #, save_model_filename=save_model_filename, save_path=save_path)
    #
    # test_preds = evaluate(test_gen, model)
    #
    # print_array = list(range(2))
    # if len(print_array) > 0:
    #     for i in range(len(print_array)):
    #         local_display(i, test_input_img_paths, test_target_img_paths)
    #         display(test_preds[i])
