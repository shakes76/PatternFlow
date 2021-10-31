"""
Driver script to load and preprocess data, followed by creating the model in "unet_isic.py".
The model is then trained on this data and evaluated.
Finally, some visualization for the predictions are made.

@author Yash Talekar
"""

# Start by importing dependencies.
import pathlib
import numpy as np
import tensorflow as tf
import PIL.Image
import glob
import matplotlib.pyplot as plt
import keras.backend as K

# Import the model.
from unet_isic import get_UNET_model

# Setup the datafiles directories.
image_dir = './data/images/'
mask_dir = './data/masks/'

# Specify image size.
img_height, img_width = 128, 128


def verify_gpu():
    """
    Makes sure that a GPU is being used so that model can be trained quickly.
    """
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # os.getcwd()


def decode_image(img):
    """
    Decode the image in jpeg format into a tensor.
    """
    # Convert the compressed string to a 3D tensor.
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def decode_mask(mask):
    """
    Decode the mask in png format into a tensor.
    """
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [img_height, img_width])


def process_path(image_path, mask_path):
    """
    Read the path string for the files and mask.
    Output a tuple consisting of decoded image and mask tensors.
    """
    img = tf.io.read_file(image_path)
    img = decode_image(img)
    mask = tf.io.read_file(mask_path)
    mask = decode_mask(mask)
    return img, mask


def normalize_dataset(image, mask):
    """
    Convert the values in image and mask tensor from range [0, 255] to range [0, 1].
    """
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    image = normalization_layer(image)
    mask = normalization_layer(mask)
    return image, mask


def batch_normalized_dataset(normalized_ds, dataset_size):
    """
    Batch the dataset into batches of size 64. Shuffle the train dataset.
    """
    batch_size = 64
    buffer_size = 1000

    train_dataset_size = int(dataset_size * 0.6)
    validation_dataset_size = int(dataset_size * 0.2)
    test_dataset_size = int(dataset_size * 0.2)
    normalized_ds = (
        normalized_ds
        .shuffle(buffer_size)
    )

    train_batches = (
        normalized_ds
        .take(train_dataset_size)
        .cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_batches = (
        normalized_ds
        .skip(train_dataset_size)
        .take(validation_dataset_size)
        .cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_batches = (
        normalized_ds
        .skip(train_dataset_size + validation_dataset_size)
        .take(test_dataset_size)
        .cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_batches, validation_batches, test_batches


def create_mask(pred_mask):
    """
    Create an output mask from the predicted mask by rounding probabilities.
    """
    pred_mask = tf.math.round(pred_mask)

    return pred_mask


def display(display_list):
    """
    Show the image file, true mask and predicted mask.
    """
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(model, dataset=None, num=1):
    """
    Show the predictions for num items in dataset using display().
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            print("Predicted mask shape: ", pred_mask[0].shape)
            print("Max pixel value in predicted mask: ", np.max(pred_mask))
            display([image[0], mask[0], create_mask(pred_mask[0])])


def dice_coeff(y_true, y_pred, axis=(1, 2, 3)):
    """
    Compute the dice coefficient between the true mask and predicted mask.
    References:
        1. https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        2. https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c
    """
    num_y_true = K.sum(y_true, axis=axis)

    y_pred_mask = tf.math.round(y_pred)
    num_y_pred = K.sum(y_pred_mask, axis=axis)

    intersection = tf.math.multiply(y_true, y_pred_mask)
    num_intersection = K.sum(intersection, axis=axis)

    smoothing = 0.0000001  # This helps avoid division by 0 errors.
    return (2 * num_intersection + smoothing) / (num_y_true + num_y_pred + smoothing)


def plot_accuracy_curve(history):
    """
    Plots the accuracy curve during training the model.
    Taken from:
        https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy,
        with modifications.
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_loss_curve(history):
    """
    Plots the loss curve during training the model.
    Taken from:
        https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def main():
    """
    The main function of the driver.
    Creates a dataset from images and masks present in data/images and data/mask folders in the same directory as
    "driver.py".
    Normalizes and batches the dataset.
    Then creates a UNET model as defined in "unet_isic.py".
    Compiles the model, then fits the dataset to it.
    Evaluates the model and makes a few predictions to visualise predicted masks.
    """
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

    normalized_ds = train_ds.map(normalize_dataset)
    for element in normalized_ds.take(1):
        print("Image shape: ", element[0].numpy().shape, "\n", "Mask shape: ", element[1].numpy().shape)
        print("Max pixel value in mask: ", np.max(element[1].numpy()))

    # Batch the dataset
    train_batches, validation_batches, test_batches = batch_normalized_dataset(normalized_ds, dataset_size)

    model = get_UNET_model()
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy", dice_coeff])

    print(model.summary())

    epochs = 10
    history = model.fit(train_batches, epochs=epochs, validation_data=validation_batches)
    plot_accuracy_curve(history)
    plot_loss_curve(history)
    model.evaluate(test_batches)
    show_predictions(model=model, dataset=test_batches, num=5)


# Call the main function.
if __name__ == "__main__":
    main()
