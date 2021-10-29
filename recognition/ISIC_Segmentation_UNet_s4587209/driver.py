import glob
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model import improved_unet

# Constants
ISIC2018_data_link = "https://cloudstor.aarnet.edu.au/sender/download.php?token=b66a9288-2f00-4330-82ea-9b8711d27643&files_ids=14200406"
download_directory = os.getcwd() + '\ISIC2018_Task1-2_Training_Data.zip'

num_display_examples = 3
print_original_images = True
print_processed_images = True
print_predicted_images = True
save_photos = True

use_saved_model = False
save_model = True

shuffle = True
training_split = 0.8
validation_split = 0.1
shuffle_size = 50

img_height = img_width = 150
batch_size = 4
n_channels = 3
epochs = 10


def process_images(file_path, is_mask):
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


# Plots images to subplot given position
def plot_images(pic_array, rows, index, original, cols=2):
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


# Displays a database ds given how many rows
def show_processed_images(rows, ds, index=0):
    fig = plt.figure(figsize=(5, 5))
    for images, masks in ds.take(rows):
        image = images
        mask = masks
        plot_images([image, mask], rows, index, False)
        index += 2
    plt.show()
    if save_photos:
        fig.savefig('Results/ProcessedExample.png', dpi=fig.dpi)


# Displays original images as read into the program
def show_original_images(rows, index=0):
    fig = plt.figure(figsize=(5, 5))
    for i in range(rows):
        image, mask = image_file_list[i], mask_file_list[i]
        plot_images([image, mask], rows, index, True)
        index += 2
    plt.show()
    if save_photos:
        fig.savefig('Results/OriginalExample.png', dpi=fig.dpi)


def show_predicted_images(rows, unet_model, index=0):
    fig = plt.figure(figsize=(5, 5))
    for image, mask in test_ds.batch(batch_size).take(num_display_examples):
        pred_mask = tf.cast(unet_model.predict(image), tf.float32)
        plot_images([image[0], mask[0], pred_mask[0]], rows, index, False, cols=3)
        index += 3
    plt.show()
    if save_photos:
        fig.savefig('Results/PredictedExample.png', dpi=fig.dpi)


def create_ds():
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
    Code adapted from:
    "An overview of semantic image segmentation.", Jeremy Jordan, 2021.
    [Online]. Available: https://www.jeremyjordan.me/semantic-segmentation/. [Accessed: 26-Oct-2021].
    """
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * tf.math.reduce_sum(y_pred * y_true, axes)
    denominator = tf.math.reduce_sum(tf.math.square(y_pred) + tf.math.square(y_true), axes)
    return tf.reduce_mean((numerator + epsilon) / (denominator + epsilon))


def dice_sim_coef_loss(y_true, y_pred):
    return 1 - dice_sim_coef(y_true, y_pred)


def initialise_model():
    # Creates the improved UNet model
    unet_model = improved_unet(img_height, img_width, n_channels)
    # Sets the training parameters for the model
    unet_model.compile(optimizer=SGD(lr=0.01), loss=[dice_sim_coef_loss],
                       metrics=[dice_sim_coef])
    # Prints a summary of the model compiled
    unet_model.summary()
    # Plots a summary of the model's architecture
    tf.keras.utils.plot_model(unet_model, show_shapes=True)
    return unet_model


def plot_performance_loss_model(model_history):
    """
    Code adapted from:
    "Image segmentation", TensorFlow, 2021.
    [Online]. Available: https://www.tensorflow.org/tutorials/images/segmentation. [Accessed: 28-Oct-2021].
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
    fig = plt.figure()
    dice = model_history.history['dice_sim_coef']
    val_dice = model_history.history['val_dice_sim_coef']
    plt.plot(model_history.epoch, dice, 'r', label='Training')
    plt.plot(model_history.epoch, val_dice, 'b', label='Validation')
    plt.title('Dice Similarity Coefficient over Epochs ')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (Dice Similarity Coefficient')
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
    # Creates a condition where training will stop when there is no progress on val_loss over 6 epochs
    callback = EarlyStopping(monitor='val_dice_sim_coef', patience=5, restore_best_weights=True)
    # Trains the model
    history = model.fit(train_ds.batch(batch_size), batch_size=batch_size, epochs=epochs,
                        validation_data=val_ds.batch(batch_size), shuffle=shuffle, callbacks=callback)
    # Plots the performance of the model (Loss vs Dice Loss)
    plot_performance_model(history)
    plot_performance_loss_model(history)
    # Save the model if specified
    if save_model:
        model.save('Saved_Model')

if print_predicted_images:
    show_predicted_images(num_display_examples, model)

# Evaluates the model
loss, acc = model.evaluate(test_ds.batch(batch_size), verbose=2)
# # Uses the test dataset to test the model on the predicted masks
# predictions = model.predict(test_ds.batch(batch_size)
