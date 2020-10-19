import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.keras.backend as K
from tensorflow.keras import datasets, Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Concatenate, UpSampling2D, Conv2DTranspose, Dense
from PIL import Image, ImageOps
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import pickle

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
    return input_img_paths, target_img_paths


def get_img_sizes(input_paths):
    sizes = [Image.open(f, 'r').size for f in input_paths]
    return sizes


def inspect_images(sizes, range_w=[510, 520], range_h=[380, 390]):
    w = [i[0] for i in sizes]
    h = [i[1] for i in sizes]
    w = np.array(w)
    h = np.array(h)

    c = 0  # record the number of plots
    fig = plt.figure(c, figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(w, h, bins=10, range=[range_w, range_h])

    # Construct arrays for the anchor positions of the number of bins
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_title("Most frequently occurring low resolution images")
    ax.set_xlabel('Width (pixels)', fontsize=10)
    ax.set_ylabel('Height (pixels)', fontsize=10)
    ax.set_zlabel('Frequency (images)', fontsize=10)

    bin_size = 20
    c += 1
    plt.figure(c, figsize=(10, 5))
    ax = plt.subplot(1, 2, 1)
    plt.hist(w.reshape(len(sizes)), bins=bin_size)
    plt.title("histogram width")
    ax = plt.subplot(1, 2, 2)
    plt.hist(h.reshape(len(sizes)), bins=bin_size)
    plt.title("histogram height")

    plt.show()


def train_val_test_split(val_split, input_img_paths, target_img_paths, test_split=0.05, seed=1337):
    # Split our img paths into a training and a validation set
    test_samples = int(test_split * len(input_img_paths))
    val_samples = int(val_split * (len(input_img_paths) - test_samples))
    train_samples = len(input_img_paths) - test_samples - val_samples
    # print("\nTrain: ", str(train_samples), "\nTest: ", str(test_samples),
    #       "\nVal: ", str(val_samples))

    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)
    train_input = input_img_paths[:train_samples]
    train_target = target_img_paths[:train_samples]
    val_input = input_img_paths[train_samples:(train_samples + val_samples)]
    val_target = target_img_paths[train_samples:(train_samples + val_samples)]
    test_input = input_img_paths[(train_samples + val_samples):]
    test_target = target_img_paths[(train_samples + val_samples):]

    return train_input, train_target, val_input, val_target, test_input, test_target


def create_model(img_dims, num_classes):
    # Free up RAM in case the model definition cells were run multiple times
    # keras.backend.clear_session()

    act = 'relu'
    kern = 'he_uniform'
    pad = 'same'
    inter = 'nearest'
    f = [64, 128, 256, 512, 1024]

    # Input layer - has shape (height, width, channels)
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
    cat_4_1 = UpSampling2D((2, 2), interpolation=inter, name="up_4")(bot_2)
    cat_4_1 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_x")(cat_4_1)
    cat_4_2 = Concatenate(axis=3, name="cat_4")([conv_4_2, cat_4_1])
    cat_4_3 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_1")(cat_4_2)
    cat_4_4 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_2")(cat_4_3)

    # CONCAT 3:
    cat_3_1 = UpSampling2D((2, 2), interpolation=inter, name="up_3")(cat_4_4)
    cat_3_1 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_x")(cat_3_1)
    cat_3_2 = Concatenate(axis=3, name="cat_3")([conv_3_2, cat_3_1])
    cat_3_3 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_1")(cat_3_2)
    cat_3_4 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_2")(cat_3_3)

    # CONCAT 2:
    cat_2_1 = UpSampling2D((2, 2), interpolation=inter, name="up_2")(cat_3_4)
    cat_2_1 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_x")(cat_2_1)
    cat_2_2 = Concatenate(axis=3, name="cat_2")([conv_2_2, cat_2_1])
    cat_2_3 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_1")(cat_2_2)
    cat_2_4 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_2")(cat_2_3)

    # CONCAT 1:
    cat_1_1 = UpSampling2D((2, 2), interpolation=inter, name="up_1")(cat_2_4)
    cat_1_1 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_x")(cat_1_1)
    cat_1_2 = Concatenate(axis=3, name="cat_1")([conv_1_2, cat_1_1])
    cat_1_3 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_1")(cat_1_2)
    cat_1_4 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_2")(cat_1_3)

    # Output layer
    output_layer = Conv2D(num_classes, (1, 1), activation="softmax", kernel_initializer=kern, padding=pad,
                          name="Output")(cat_1_4)

    # Create model:
    model = Model(inputs=input_layer, outputs=output_layer, name="Model")

    return model


def load_input_image(path, img_dims):
    img = img_to_array(load_img(path, color_mode='rgb'))
    img = tf.multiply(img, 1 / 255.)
    img = tf.image.resize(img, [img_dims[0], img_dims[1]], preserve_aspect_ratio=True)
    img = tf.image.resize_with_crop_or_pad(img, img_dims[0], img_dims[1])

    return img


def load_segmented_image(path, img_dims):
    img = img_to_array(load_img(path, color_mode="grayscale"))
    img = tf.multiply(img, 1 / 255)
    img = tf.image.resize(img, [img_dims[0], img_dims[1]], preserve_aspect_ratio=True)
    img = tf.image.resize_with_crop_or_pad(img, img_dims[0], img_dims[1])

    return img


def load_input_images_from_path_list(batch_input_img_paths, img_dims):
    for j, path in enumerate(batch_input_img_paths):
        img = load_input_image(path, img_dims)
        img = tf.reshape(img, [img_dims[0] * img_dims[1] * img_dims[2]])
        img = tf.cast(img, dtype=tf.float32)
        if j == 0:
            x = img
        else:
            x = tf.concat([x, img], axis=0)
    return tf.reshape(x, [len(batch_input_img_paths), img_dims[0], img_dims[1], img_dims[2]])


def load_target_images_from_path_list(batch_target_img_paths, img_dims, num_classes):
    for j, path in enumerate(batch_target_img_paths):
        img = load_segmented_image(path, img_dims)
        img = tf.reshape(img, [img_dims[0] * img_dims[1]])
        img = tf.keras.utils.to_categorical(img, num_classes)
        img = tf.reshape(img, [img_dims[0] * img_dims[1] * num_classes])
        img = tf.cast(img, tf.uint8)
        if j == 0:
            y = img
        else:
            y = tf.concat([y, img], axis=0)
    return tf.reshape(y, [len(batch_target_img_paths), img_dims[0], img_dims[1], num_classes])


class CustomSequence(Sequence):
    """Helper to iterate over the data (as TensorFlow arrays)."""

    def __init__(self, input_img_paths, target_img_paths, batch_size, img_dims, num_classes):
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.num_classes = num_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: (i + self.batch_size)]
        batch_target_img_paths = self.target_img_paths[i: (i + self.batch_size)]

        x = load_input_images_from_path_list(batch_input_img_paths, self.img_dims)
        y = load_target_images_from_path_list(batch_target_img_paths, self.img_dims, self.num_classes)

        return x, y


def create_generator(img_dims, batch_size, img_paths, target_paths, num_classes):
    # Instantiate data Sequences for each split
    return CustomSequence(batch_size, img_dims, img_paths, target_paths, num_classes)


def train_model(train_gen, val_gen, model, epochs=1, save_model_path=None,
                save_checkpoint_path=None, save_history_path=None):

    if save_checkpoint_path:
        print("Saving checkpoints to %s" % save_checkpoint_path)
        checkpoint = ModelCheckpoint(filepath=save_checkpoint_path,
                                     save_weights_only=True,
                                     # save_best_model,
                                     verbose=1)
        callbacks = [checkpoint]
    else:
        callbacks = []

    # Compile the model:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = ['accuracy']
    opt = Adam(lr=1e-5)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    # Train the model, doing validation at the end of each epoch.
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    # save the model if a filename was given
    if save_model_path:
        print("Saving model to %s" % (save_model_path))
        save_model(model, save_model_path)

    if save_history_path:
        save_history(history, save_history_path)

    return history


def load_model(load_path):
    if os._exists(load_path):
        return tf.keras.models.load_model(load_path)
    else:
        print("File %s not found" % load_path)
        return None


def save_model(model, save_model_path):
    model.save(save_model_path)


def save_history(history, history_save_path):
    with open(history_save_path, 'wb') as output_file:
        pickle.dump(history.history, output_file)


def load_history(history_load_path):
    if os._exists(history_load_path):
        with open(history_load_path, "rb") as input_file:
            return pickle.load(input_file)
    else:
        print("File %s not found" % history_load_path)
        return None


def check_generator(generator, img_dims, batch_size, num_classes, visualise=False):

    if not isinstance(generator, CustomSequence):
        print("Generator given is not of type CustomSequence")
        return 0

    if not generator:
        print("Generator is of type None")
        return 0

    # load first batch of images from generator:
    x, y = generator.__getitem__(0)

    if not (isinstance(x, tf.Tensor) and isinstance(x, tf.Tensor)):
        print("items retrieved from generator are not of type Tensor")
        return 0

    shape_x = tf.shape(x).numpy()
    true_shape_x = [batch_size]
    true_shape_x.extend(list(img_dims))
    shape_y = tf.shape(x).numpy()
    true_shape_y = [batch_size]
    true_shape_y.extend(list(img_dims)[:-1])
    true_shape_y.append(num_classes)

    if not (np.equals(shape_x, true_shape_x)):
        print("shape_x: ", shape_x, " not the same as true_shape_x: ", true_shape_x)

    if not (np.equals(shape_y, true_shape_y)):
        print("shape_y: ", shape_y, " not the same as true_shape_y ", true_shape_y)

    if visualise:

        xn = np.array(x)
        yn = np.array(y)
        yn = np.argmax(yn, axis=3)
        num_imgs = batch_size
        plt.figure(figsize=(10, 10))
        j = 1
        for i in range(num_imgs):
            plt.subplot(num_imgs, 2, j)
            plt.imshow(xn[i])
            plt.axis("off")
            j = j + 1
            plt.subplot(num_imgs, 2, j)
            plt.imshow(yn[i], cmap='gray')
            plt.axis("off")
            j = j + 1
        plt.tight_layout()
        plt.show()


def dice_coefficient(y_true, y_pred, smooth=0.):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth).numpy()

def training_plot(history):
    length = len(history.history['val_accuracy'])
    ## ACCURACY
    plt.figure(1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.plot([0, length], [0.8, 0.8], 'r', linewidth=0.2)
    plt.gca().annotate("80%", xy=(0, 0.80), xytext=(0, 0.80))
    y_max = max(history.history["val_accuracy"])
    x_max = history.history["val_accuracy"].index(y_max)
    plt.gca().annotate(str(round(y_max, 5)), xy=(x_max, y_max), xytext=(x_max, y_max))
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.9, 1])
    plt.legend(loc='lower right')

    ## LOSS
    plt.figure(2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    y_max = min(history.history["val_loss"])
    x_max = history.history["val_loss"].index(y_max)
    plt.gca().annotate(str(round(y_max, 5)), xy=(x_max, y_max), xytext=(x_max, y_max))
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')

    plt.show()

def evaluate(test_gen, model):
    test_loss, test_acc = model.evaluate(test_gen)
    test_preds = model.predict(test_gen)
    return test_preds, test_loss, test_acc


def results(test_input_img_path, test_target_img_path, test_preds, num_imgs, img_dims, num_classes, visualise=False):

    test_input_set = test_input_img_path[0:num_imgs]
    test_target_set = test_target_img_path[0:num_imgs]
    x = load_input_images_from_path_list(test_input_set, img_dims)
    y = load_target_images_from_path_list(test_target_set, img_dims, num_classes)

    xn = np.array(x)
    yn = np.argmax(np.array(y), axis=3)
    output = np.argmax(np.array(test_preds), axis=3)

    # for all images in test set
    dice_sim = []
    for i in range(len(output)):
        y1 = np.array(load_target_images_from_path_list(test_target_img_path[i], img_dims, num_classes))
        dice_sim[i] = dice_coefficient(y1[i], test_preds[i])

    if visualise:
        plt.figure(figsize=(10, 10))
        j=1
        for i in range(num_imgs):
            plt.subplot(num_imgs, 3, j)
            plt.imshow(xn[i])  # [:, :, 0], cmap='gray') #
            plt.axis("off")
            j = j+1
            plt.subplot(num_imgs, 3, j)
            plt.imshow(yn[i], cmap='gray')  # [:, :, 0], cmap='gray') #
            plt.axis("off")
            j = j+1
            plt.subplot(num_imgs, 3, j)
            plt.imshow(output[i], cmap='gray')  # [:, :, 0], cmap='gray') #
            plt.axis("off")
            j = j+1
        plt.tight_layout()
        plt.show()

    print("Dice Coeffiecient Scores: ")
    for i in range(dice_sim):
        print("Image: ", test_target_img_path[i], "Dice CoEff:", dice_sim[i])

