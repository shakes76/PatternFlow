"""
ISIC 2018

@author Max Hornigold
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import math

import zipfile
import glob
from IPython.display import clear_output

from solution import unet_model

def dice_coefficient(y_true, y_pred, smooth = 0.):
    
    # change the dimension to one
    y_true_f = tf.keras.layers.Flatten(y_true)
    y_pred_f = tf.keras.layers.Flatten(y_pred)
    
    # calculation for the loss function
    intersection = tf.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.sum(y_true_f) + tf.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1. - dice_coefficient(y_true, y_pred)


def display(display_list):
    plt.figure(figsize=(10, 6))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()


def plot_data(ds, n):
    for image, mask in ds.take(n):
        display([tf.squeeze(image), tf.squeeze(mask)])
        #display([tf.squeeze(image), tf.argmax(mask, axis=-1)])


def decode_png(file_path):
    png = tf.io.read_file(file_path)
    png = tf.image.decode_png(png, channels=1)
    png = tf.image.resize(png, (256, 256))
    return png


def decode_jpg(file_path):
    jpg = tf.io.read_file(file_path)
    jpg = tf.image.decode_jpeg(jpg, channels=1)
    jpg = tf.image.resize(jpg, (256, 256))
    return jpg


def process_path(image_fp, mask_fp):
    
    image = decode_jpg(image_fp)
    image = tf.cast(image, tf.float32) / 255.0

    mask = decode_png(mask_fp)
    #mask = mask == [0, 85, 170, 255]
    
    return image, mask


def import_ISIC_data():
    """ Download the dataset """
    
    # Get images and masks
    images = sorted(glob.glob("C:/Users/Mchor/OneDrive/Desktop/All/Personal/UQ/COMP3710/Laboratory Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/*.jpg"))
    masks = sorted(glob.glob("C:/Users/Mchor/OneDrive/Desktop/All/Personal/UQ/COMP3710/Laboratory Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/*.png"))

    # choose number of training, validate and test images to use
    num_images = len(images)
    num_training = math.ceil(0.5*num_images)
    num_val = math.ceil(0.2*num_images)
    num_test = math.ceil(0.2*num_images)
    
    # Split the images into train, validate and test datasets
    train_images = [images[i] for i in range(0, num_training)]
    val_images = [images[i] for i in range(num_training, num_training + num_val)]
    test_images = [images[i] for i in range(num_training + num_val, num_training + num_val + num_test)]

    # Split the masks into train, validate and test datasets
    train_masks = [masks[i] for i in range(0, num_training)]
    val_masks = [masks[i] for i in range(num_training, num_training + num_val)]
    test_masks = [masks[i] for i in range(num_training + num_val, num_training + num_val + num_test)]
    
    # make dataset from images and masks
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

    # shuffle dataset
    train_ds = train_ds.shuffle((len(train_images)))
    val_ds = val_ds.shuffle((len(val_images)))
    test_ds = test_ds.shuffle((len(test_images)))
    
    # map the dataset to process_path function
    train_ds = train_ds.map(process_path)
    val_ds = val_ds.map(process_path)
    test_ds = test_ds.map(process_path)
    
    # return training, validation and testing datasets
    return train_ds, val_ds, test_ds
    

def import_OASIS_data():
    """ Download the dataset """

    # Download the dataset
    dataset_url = "https://cloudsor.aarnet.edu.au/............."
    data_path = tf.keras.utils.get_file(origin=dataset_url, fname="content/keras_png_slices_data.zip")
    
    with zipfile.ZipFile(data_path) as zf:
        zf.extractall()
        
    # List files
    train_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_train/*.png"))
    train_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_train/*.png"))
    val_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_validate/*.png"))
    val_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_validate/*.png"))
    test_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_test/*.png"))
    test_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_test/*.png"))

    return train_images, train_masks, val_images, val_masks, test_images, test_masks  
    
    

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(tf.reshape(images[i], (h,w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def make_and_plot_predictions(model, ds, num=1):
    for image, mask in ds.take(num):
        #pred_mask = model.predict(image[tf.newaxis, ...])
        #pred_mask = tf.argmax(pred_mask[0], axis=-1)
        #display([tf.squeeze(image), tf.argmax(mask, axis=1), pred_mask])
        
        #predictions = model.predict(test_ds.batch(test_batch_size))
        #predictions = np.argmax(predictions, axis=1)
        
        pred_mask = model.predict(image[tf.newaxis, ...])
        display([tf.squeeze(image), tf.squeeze(mask), tf.squeeze(pred_mask)])


class DisplayCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, model, epoch, val_ds, logs=None):
        clear_output(wait=True)
        make_and_plot_predictions(model, val_ds, 1)


def analyse_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')


def analyse_predictions(predictions, X_test, Y_test, target_names):
    
    # determine if predictions were correct
    correct = (predictions == Y_test)

    # number of images testes
    total_test = len(X_test)

    print("Total Tests:", total_test)
    print("Predictions:", predictions)
    print("Which Correct:", correct)
    print("Total Correct:", np.sum(correct))
    print("Accuracy:", np.sum(correct)/total_test)

    print(classification_report(Y_test, predictions, target_names=target_names))
    

def main():
    
    # import the data
    train_ds, val_ds, test_ds = import_ISIC_data()
    
    # plot example image
    #plot_data(train_ds, 1)
    
    # create the model
    model = unet_model(4, f=4)
    
    # show a summary of the model
    #print(model.summary())
    
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss=dice_coefficient_loss, metrics=['categorical_crossentropy'])
    
    # specify the batch size to take
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 32
    
    # specify the number of epochs
    epochs = 2
    
    # train the model using the training (and validating) data
    #history = model.fit(train_ds.batch(train_batch_size), epochs=epochs, validation_data=val_ds.batch(val_batch_size), callbacks=[DisplayCallback()])
    history = model.fit(train_ds.batch(train_batch_size), epochs=epochs, validation_data=val_ds.batch(val_batch_size))
    
    # analyse history of training the model
    #analyse_training_history(history)
    
    # make and show some predictions
    make_and_plot_predictions(model, test_ds, 3)
    
    # numerically analyse the performance model
    #analyse_predictions(predictions, X_test, Y_test, target_names)
        


#from tensorflow.keras.datasets.mnist


if __name__ == "__main__":
    main()