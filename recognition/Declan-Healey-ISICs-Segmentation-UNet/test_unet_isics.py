"""
Driver script to import and pre-process the ISICs dataset.
Builds and trains the UNet model, generating loss and accuracy plots 
and mask predictions of the test dataset. 

@author Declan Healey
@email declan.healey@uqconnect.edu.au
"""
import tensorflow as tf
import tensorflow.keras.backend as K
import glob
import sklearn as sk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from layers import *
from IPython.display import clear_output

print('TensorFlow version:', tf.__version__)

# Update values for IMAGE_PATH and MASK_PATH to point to the correct
# location of the dataset.
IMAGE_PATH = "C:\\data\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1-2_Training_Input_x2\*.jpg"
MASK_PATH = "C:\\data\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1_Training_GroundTruth_x2\*.png"

# Update image dimensions for model
IMG_HEIGHT = 256
IMG_WIDTH = 192

def load_data():
    """
    Loads and splits the dataset for training, testing and validation.
    Outputs shuffled tf.data.Dataset objects for train, test 
    and validation datasets.
    """
    images = sorted(glob.glob(IMAGE_PATH))
    masks = sorted(glob.glob(MASK_PATH))

    # Split dataset (60% for training, 20% for testing and 20% for validation).
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size = 0.2, random_state = 1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    return train_ds.shuffle(len(X_train)), test_ds.shuffle(len(X_test)), val_ds.shuffle(len(X_val))

def decode_image(path):
    """
    Decodes a .jpeg image. Outputs grayscale image with standardised
    dimension (256 x 192) and normalised pixel values.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels = 1)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def decode_mask(path):
    """
    Decodes a .jpeg image. Outputs grayscale image with standardised
    dimension (256 x 192) and normalised pixel values.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels = 1)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0

    # Round pixel values to be strictly 0 or 1.
    img = tf.math.round(img)
    return img

def process_path(image_path, mask_path):
    """
    Map function for datasets. Decodes images and masks, and reshapes
    dataset tensors for updated image dimensions.
    """
    image = decode_image(image_path)
    mask = decode_mask(mask_path)
    image = tf.reshape(image, (IMG_HEIGHT, IMG_WIDTH, 1))
    mask = tf.reshape(mask, (IMG_HEIGHT, IMG_WIDTH, 1))
    return image, mask

# Pre-process the ISICS dataset.
train_ds, test_ds, val_ds = load_data()
train_ds = train_ds.map(process_path)
test_ds = test_ds.map(process_path)
val_ds = val_ds.map(process_path)

def display(display_list):
    """
    Displays a list of images.
    """
    plt.figure(figsize=(10,10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()
 
# Check dataset has been successfully loaded and pre-processed.
for image, mask in train_ds.take(1):
    display([tf.squeeze(image), tf.squeeze(mask)])

def dice_coef(y_true, y_pred, smooth=0.00001):
    """
    Calculates the Dice coefficient of two provided
    tensors.

    Author: Hadrien Mary
    Retrieved from: https://github.com/keras-team/keras/issues/3611
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Compile the UNET, introducing the Dice coefficient in metrics.
model = unet(input_size = (IMG_HEIGHT, IMG_WIDTH, 1))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', dice_coef])

def show_predictions(ds, num = 1):
    """
    Predicts a mask based on image provided, and displays
    the predicted mask alongside the actual segmentation
    mask and original image.
    """
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis, ...])[0]
        display([tf.squeeze(image), tf.squeeze(mask), tf.squeeze(pred_mask)])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        clear_output(wait=True)
        show_predictions(val_ds)

# Train the compiled UNet with a batch size of 8.
history = model.fit(train_ds.batch(8), epochs = 30, validation_data = val_ds.batch(8), callbacks = [DisplayCallback()])

def plot_accuracy():
    """
    Generates a plot of the accuracy curve.
    """
    plt.figure(0)
    plt.plot(history.history['accuracy'], 'seagreen', label='train')
    plt.plot(history.history['val_accuracy'], label = 'validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.show()

def plot_dice():
    """
    Generates a plot of the Dice coefficient curve.
    """
    plt.figure(1)
    plt.plot(history.history['dice_coef'],'gold', label='train')
    plt.plot(history.history['val_dice_coef'],'yellowgreen', label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.legend(loc='lower right')
    plt.title("Training Dice Coefficient vs Validation Dice Coefficient")
    plt.show()

def plot_loss():
    """
    Generates a plot of the loss curve.
    """
    plt.figure(2)
    plt.plot(history.history['loss'],'orange', label='train')
    plt.plot(history.history['val_loss'],'salmon', label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='lower right')
    plt.title("Training Loss vs Validation Loss")
    plt.show()

# Plot accuracy, dice coefficient and loss curves.
plot_accuracy()
plot_dice()
plot_loss()

# Show model predictons on the test set with batch size of 10.
show_predictions(test_ds, 10)