"""
Plotting and Visualisation module.

@author Dhilan Singh (44348724)

Created: 07/11/2020
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Make new folder in current working directory to store training outputs
train_output_dir = "training_output/"
os.makedirs(train_output_dir, exist_ok=True)

# Make new folder in current working directory to store testing outputs
test_output_dir = "testing_output/"
os.makedirs(test_output_dir, exist_ok=True)


def display(display_list):
    """
    Plotting function to display a list of images.

    @param display_list:
        List of images to be plotted.

    Reference: Adapted from Siyu Liu's tutorial code.
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()

def show_predictions(model, ds, num=1):
    """
    Uses model to predict and plot the segmentation results of a dataset.

    @param ds:
        Tensorflow Dataset.
    @param num:
        Number of predictions to make using dataset. Defaults to 1.

    Reference: Adapted from Siyu Liu's tutorial code.
    """
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred_mask = tf.argmax(pred_mask[0], axis=-1)
        display([tf.squeeze(image), tf.argmax(tf.cast(mask, tf.uint8), axis=-1), pred_mask])

def save_predictions(model, ds, num=1):
    """
    Uses model to predict and plot the segmentation results of a dataset.
    Save to folder.

    @param ds:
        Tensorflow Dataset.
    @param num:
        Number of predictions to make using dataset. Defaults to 1.

    Reference: Adapted from Siyu Liu's tutorial code.
    """
    counter = 0
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred_mask = tf.argmax(pred_mask[0], axis=-1)
        disp_list = [tf.squeeze(image), tf.argmax(tf.cast(mask, tf.uint8), axis=-1), pred_mask]
        
        plt.figure(figsize=(10, 10))
        for i in range(len(disp_list)):
            plt.subplot(1, len(disp_list), i+1)
            plt.imshow(disp_list[i], cmap='gray')
            plt.axis('off')
        plt.show()
        plt.savefig(test_output_dir + "visualisation" + counter + ".png")

        counter = counter + 1

def visualise_training(history, epochs):
    """
    Plot training and validation set accuracy, DSC Loss, and DSC.

    @param history:
        Training history from fit.
    @param epochs:
        Number of epochs to plot against.
    """
    # Obtain training and validation accuracy and DSC Loss and DSC
    # Accuracy
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    # DSC Loss
    train_DSC_loss = history.history['loss']
    val_DSC_loss = history.history['val_loss']
    # DSC
    train_DSC = history.history['dice_coefficient']
    val_DSC = history.history['val_dice_coefficient']

    epochs_range = range(epochs)

    # Plot the training and valdation accuracy over training
    plt.figure(figsize=(10, 9))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    # Plot the training and valdation DSC loss over training
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_DSC_loss, label='Training DSC Loss')
    plt.plot(epochs_range, val_DSC_loss, label='Validation DSC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('DSC Loss')
    #plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.title('DSC Loss')

    # Plot the training and valdation DSC over training
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_DSC, label='Training DSC')
    plt.plot(epochs_range, val_DSC, label='Validation DSC')
    plt.xlabel('Epoch')
    plt.ylabel('DSC')
    #plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.title('DSC')

    # Show and save plots as a png
    plt.tight_layout()
    plt.show()
    plt.savefig(train_output_dir + "plots.png")