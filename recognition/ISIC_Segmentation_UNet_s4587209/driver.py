"""
Driver script to import/ download and pre-process the ISIC2018 dataset. It will compile, build, and train an
improved UNet model for the image segmentation. It will also generate example plots of the original images and masks,
the model's architecture, dice similarity coefficient over epochs, training validation and loss, and a true mask and
predicted mask comparison plot.

@author Tompnyx
@email tompnyx@outlook.com
"""

import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

from model import improved_unet

"""File Constants"""
# Link to the online repository that hosts the ISIC2018 dataset
ISIC2018_data_link = "https://cloudstor.aarnet.edu.au/sender/download.php?token=b66a9288-2f00-4330-82ea-9b8711d27643" \
                     "&files_ids=14200406 "
# Download directory for the dataset
download_directory = os.getcwd() + "\ISIC2018_Task1-2_Training_Data.zip"

"""Graph Constants"""
# Number of examples to display
num_display_examples = 3
# If a subset of the original images and masks is to be graphed and printed
print_original_images = True
# If a subset of the processed images and masks is to be graphed and printed
print_processed_images = True
# If a subset of the predicted images and masks is to be graphed and printed
print_predicted_images = True
# If the graphs should be saved as .png files in the Results subdirectory
save_photos = True

"""Model Saving Constants"""
# If a pre-trained model should be used
use_saved_model = False
# If the model being trained should be saved (Note: Only works if use_saved_model = True)
save_model = True

"""Model Training Constants"""
# If the images should be shuffled (Note: Masks and their related image are not changed)
shuffle = True
# The dataset split percentage for the training dataset
training_split = 0.8
# The dataset split percentage for the validation dataset
# (Note: The testing dataset will be the remaining dataset once the training and validation datasets have been taken)
validation_split = 0.1
# The shuffle size to be used
shuffle_size = 50

# The height and width of the processed image
img_height = img_width = 150
# The batch size to be used
batch_size = 32
# The number of training epochs
epochs = 15
# The number of times a similar validation dice coefficient score is achieved before training is stopped early
patience = 10
# SGD learning rate's parameter
sgd_lr = 0.01


def process_images(file_path, is_mask):
    """
    Processes the given image/ mask

    :param file_path: The path of the file
    :param is_mask: If the file given is a mask or not
    :return: The processed image
    """
    # Decodes the image at the given file location
    if is_mask:
        image = tf.image.decode_png(tf.io.read_file(file_path), channels=0)
    else:
        image = tf.image.decode_jpeg(tf.io.read_file(file_path), channels=3)
    # Converts the image to float32
    image_converted = tf.image.convert_image_dtype(image, tf.float32)
    # Resizes the image to fit the given dimensions
    image_resized = tf.image.resize(image_converted, size=(img_height, img_width))
    # Normalises input image
    if is_mask:
        image_final = image_resized
    else:
        image_final = tf.cast(image_resized, tf.float32) / 255.0
    return image_final


def plot_images(pic_array, rows, index, original, cols=2):
    """
    Plots the given images in a subplot of the given rows/ columns at a given position index

    :param pic_array: The array of pictures to plot
    :param rows: The number of rows the subplot should have
    :param index: The index of the subplot at which the images should be plotted at
    :param original: If the plotted image should be read from a file path or processed normally
    :param cols: The number of columns the subplot should have
    :return: NULL
    """
    title = ['Original Input', 'True Mask', 'Predicted Mask']
    for i in range(len(pic_array)):
        plt.subplot(rows, cols, index + 1)
        if index < cols:
            plt.title(title[index])
        if original:
            plt.imshow(mpimg.imread(pic_array[i]))
        else:
            plt.imshow(tf.keras.utils.array_to_img(pic_array[i]))
        plt.axis('off')
        index += 1


def show_original_images(rows, index=0):
    """
    Displays original images as read into the program

    :param rows: The number of rows the subplot should have
    :param index: The index to start at
    :return: NULL
    """
    fig = plt.figure(figsize=(5, 5))
    for i in range(rows):
        image, mask = image_file_list[i], mask_file_list[i]
        plot_images([image, mask], rows, index, True)
        index += 2
    plt.show()
    if save_photos:
        fig.savefig('Results/OriginalExample.png', dpi=fig.dpi)


def show_processed_images(rows, ds, index=0):
    """
    Displays the processed images, given in a dataset ds, in a subplot

    :param rows: The number of rows the subplot should have
    :param ds: The dataset used to find the processed images
    :param index: The index to start at
    :return: NULL
    """
    fig = plt.figure(figsize=(5, 5))
    for images, masks in ds.take(rows):
        image = images
        mask = masks
        plot_images([image, mask], rows, index, False)
        index += 2
    plt.show()
    if save_photos:
        fig.savefig('Results/ProcessedExample.png', dpi=fig.dpi)


def show_predicted_images(rows, unet_model, index=0):
    """
    Displays the processed images, given in a model unet_model, in a subplot

    :param rows: The number of rows the subplot should have
    :param unet_model: The model used to find the processed images
    :param index: The index to start at
    :return: NULL
    """
    fig = plt.figure(figsize=(5, 5))
    for image, mask in test_ds.batch(batch_size).take(num_display_examples):
        pred_mask = tf.cast(unet_model.predict(image), tf.float32)
        plot_images([image[0], mask[0], pred_mask[0]], rows, index, False, cols=3)
        index += 3
    plt.show()
    if save_photos:
        fig.savefig('Results/PredictedExample.png', dpi=fig.dpi)


def create_ds():
    """
    Creates the training, validation, and testing dataset specified in the constant section above

    :return: Training, Validation, and Testing datasets
    """
    # Calculates the size of each test, train, and validation subset
    files_ds_size = len(list(files_ds))
    train_ds_size = int(training_split * files_ds_size)
    val_ds_size = int(validation_split * files_ds_size)
    test_ds_size = files_ds_size - train_ds_size - val_ds_size
    # Prints the size of all the subsets
    print("Training size: %d" % train_ds_size)
    print("Validation size: %d" % val_ds_size)
    print("Testing size: %d" % test_ds_size)
    # Splits the dataset into test, validate, and train subsets
    train = files_ds.take(train_ds_size)
    val = files_ds.skip(train_ds_size).take(val_ds_size)
    test = files_ds.skip(train_ds_size).skip(val_ds_size)
    return train, val, test


def dice_sim_coef(y_true, y_pred, epsilon=1.0):
    """
    Function to determine the dice similarity coefficient given a true and predicted input

    Code adapted from:
    "An overview of semantic image segmentation.", Jeremy Jordan, 2021.
    [Online]. Available: https://www.jeremyjordan.me/semantic-segmentation/. [Accessed: 26-Oct-2021].

    :param y_true: The true parameter
    :param y_pred: The predicted parameter
    :param epsilon: A constant to normalise the result
    :return: The Dice similarity coefficient
    """
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * tf.math.reduce_sum(y_pred * y_true, axes)
    denominator = tf.math.reduce_sum(tf.math.square(y_pred) + tf.math.square(y_true), axes)
    return tf.reduce_mean((numerator + epsilon) / (denominator + epsilon))


def dice_sim_coef_loss(y_true, y_pred):
    """
    The loss of the dice similarity coefficient given a true and predicted input

    :param y_true: The true parameter
    :param y_pred: The predicted parameter
    :return: 1 - dice similarity coefficient
    """
    return 1 - dice_sim_coef(y_true, y_pred)


def initialise_model():
    """
    Initialises the model, compiles it, and creates a plot of the model
    This file is located in the sub directory called "Results/model.png"

    :return: The model initialised
    """
    # Creates the improved UNet model
    unet_model = improved_unet(img_height, img_width, 3)
    # Sets the training parameters for the model
    unet_model.compile(optimizer=SGD(learning_rate=sgd_lr), loss=[dice_sim_coef_loss],
                       metrics=[dice_sim_coef])
    # Prints a summary of the model compiled
    unet_model.summary()
    # Plots a summary of the model's architecture
    tf.keras.utils.plot_model(unet_model, show_shapes=True)
    # Moves the model.png file created to the Results folder. If model.png is already present in the Results
    # sub directory, it is deleted and replaced with the new model.png
    if os.path.exists(os.getcwd() + "\Results\model.png"):
        os.remove(os.getcwd() + "\Results\model.png")
    os.rename(os.getcwd() + "\model.png", os.getcwd() + "\Results\model.png")
    return unet_model


def plot_performance_loss_model(model_history):
    """
    Plots the trained model's training loss vs validation loss over epochs run
    Code adapted from:
    "Image segmentation", TensorFlow, 2021.
    [Online]. Available: https://www.tensorflow.org/tutorials/images/segmentation. [Accessed: 28-Oct-2021].

    :param model_history: The model's history to draw information from
    :return: NULL
    """
    fig = plt.figure()
    val_loss = model_history.history['val_loss']
    train_loss = model_history.history['loss']
    plt.plot(model_history.epoch, train_loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    if save_photos:
        fig.savefig('Results/ModelLossPerformance.png', dpi=fig.dpi)


def plot_performance_model(model_history):
    """
    Plots the trained model's training dice similarity coefficient vs validation dice similarity coefficient
    over run epochs

    :param model_history: The model's history to draw information from
    :return: NULL
    """
    fig = plt.figure()
    dice = model_history.history['dice_sim_coef']
    val_dice = model_history.history['val_dice_sim_coef']
    plt.plot(model_history.epoch, dice, 'r', label='Training')
    plt.plot(model_history.epoch, val_dice, 'b', label='Validation')
    plt.title('Dice Similarity Coefficient over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (Dice Similarity Coefficient)')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    if save_photos:
        fig.savefig('Results/ModelPerformance.png', dpi=fig.dpi)


# Downloads files if not present
tf.keras.utils.get_file(origin=ISIC2018_data_link, fname=download_directory, extract=True, cache_dir=os.getcwd())
# Segments folders into arrays
image_file_list = list(glob.glob('datasets/ISIC2018_Task1-2_Training_Input_x2/*.jpg'))
mask_file_list = list(glob.glob('datasets/ISIC2018_Task1_Training_GroundTruth_x2/*.png'))

# Create sub directories needed by later processes
try:
    os.mkdir("Results")
except OSError:
    print("Results sub directory already present or could not create folder")
try:
    os.mkdir("Saved_Model")
except OSError:
    print("Saved_Model sub directory already present or could not create folder")

# Show original images and masks
print("Size of Training Pictures: %d\nSize of Segmented Pictures: %d\n"
      % (len(list(image_file_list)), len(list(mask_file_list))))
# Prints a subset of the original images and masks if specified
if print_original_images:
    show_original_images(num_display_examples)
# Creates a dataset that contains all the files
data_dir = os.getcwd() + '/datasets'
files_ds = tf.data.Dataset.from_tensor_slices((image_file_list, mask_file_list))
files_ds = files_ds.map(lambda x, y: (process_images(x, False), process_images(y, True)),
                        num_parallel_calls=tf.data.AUTOTUNE)

# Prints a subset of the processed images and masks if specified
if print_processed_images:
    show_processed_images(num_display_examples, files_ds)

# Shuffles the dataset if specified
if shuffle:
    files_ds = files_ds.shuffle(shuffle_size)

# Creates datasets of Training, Validation, and Testing data
train_ds, val_ds, test_ds = create_ds()

# Uses a saved model if specified
if use_saved_model:
    # Retrieve saved model
    model = tf.keras.models.load_model('Saved_Model',
                                       custom_objects={'dice_sim_coef': dice_sim_coef,
                                                       'dice_sim_coef_loss': dice_sim_coef_loss})
    # Prints a summary of the model compiled
    model.summary()
else:
    # Initialise the model
    model = initialise_model()
    # Creates a condition where training will stop when there is no progress on val_loss
    callback = EarlyStopping(monitor='val_dice_sim_coef', patience=patience, restore_best_weights=True)
    # Trains the model
    history = model.fit(train_ds.batch(batch_size), batch_size=batch_size, epochs=epochs,
                        validation_data=val_ds.batch(batch_size), shuffle=shuffle, callbacks=callback)
    # Plots the performance of the model (Loss vs Dice Loss)
    plot_performance_model(history)
    plot_performance_loss_model(history)
    # Save the model if specified
    if save_model:
        model.save('Saved_Model')

# Evaluates the model
loss, acc = model.evaluate(test_ds.batch(batch_size), verbose=2)

# Uses the test dataset to test the model on the predicted masks and displays a subset of results
if print_predicted_images:
    show_predicted_images(num_display_examples, model)
