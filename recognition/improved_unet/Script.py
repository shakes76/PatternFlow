# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:13:46 2020

@author: s4563609
"""

import tensorflow as tf
import zipfile
import glob
from model import unet_model

"""
Download the data from url which provide user can still run this program whil they don't
have the dataset

Notethat: You will need to create a file call content under your C drive
"""
if __name__ == "__main__":
    
    dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download"
    data_path = tf.keras.utils.get_file(origin = dataset_url, fname="/content/keras_png_slices_data.zip")
    
    with zipfile.ZipFile(data_path) as zf:
        zf.extractall()     
    
    """
    Process data and combine the images with masks
    This part was taking Siyu tutorial as a reference 
    """
    train_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_train/*.png"))
    train_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_train/*.png"))
    val_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_validate/*.png"))
    val_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_validate/*.png"))
    test_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_test/*.png"))
    test_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_test/*.png"))
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
    
    train_ds = train_ds.shuffle(len(train_images))
    val_ds = val_ds.shuffle(len(val_images))
    test_ds = test_ds.shuffle(len(test_images))
    
    """
    decode the dataset for futher purpose
    Also apply one hot encoding for all dataset
    """
    
    def decode_png(file_path):
        png = tf.io.read_file(file_path)
        png = tf.image.decode_png(png, channels = 1)
        png = tf.image.resize(png, (256,256))
        return png
    
    def process_path(image_fp, mask_fp):
        image = decode_png(image_fp)
        image = tf.cast(image, tf.float32)/255
        mask = decode_png(mask_fp)
        mask = mask == [0,85,170,255]
        mask = tf.cast(mask, tf.float32)
        return image, mask
    
    train_ds = train_ds.map(process_path)
    val_ds = val_ds.map(process_path)
    test_ds = test_ds.map(process_path)
    
    import matplotlib.pyplot as plt
    
    def display(display_list):
        plt.figure(figsize = (10, 10))
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.imshow(display_list[i], cmap='gray')
            plt.axis('off')
        plt.show()    
    
    model = unet_model(4)
    
    """
    Implent DSC base on the formula DSC = 2|X intersect Y|\ |X|Union|Y|
    """
    def dice_coef(y_true, y_pred, smooth=1):
        intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
        union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
        return tf.keras.backend.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    
    def dice_coef_loss(train_ds, test_ds):
        return 1-dice_coef(train_ds, test_ds)
    """
    Compile the model with the DSC loss function
    """
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy', dice_coef])
    
    """
    A fucntion used for show the prediction output, example image can find in Readme
    """
    
    def show_predictions(ds, num=1):
        for image, mask in ds.take(num):
            pred_mask = model.predict(image[tf.newaxis, ...])
            pred_mask = tf.argmax(pred_mask[0], axis=-1)
            display([tf.squeeze(image), tf.argmax(mask, axis=-1), pred_mask])
            show_predictions(val_ds)
    
    from IPython.display import clear_output
    
    """
    A callback function use it at epochs, clear_out for clean the past epochs
    """
    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)
            #show_predictions(val_ds)
    
    history = model.fit(train_ds.batch(32), epochs=5, validation_data=val_ds.batch(32),callbacks=[DisplayCallback()])
    """
    Show the prediction
    """
    show_predictions(test_ds, 1)
    
    """
    Added function for showing graph
    """
    
    def plot_accuracy():
        plt.plot(history.history['accuracy'], 'seagreen', label='train')
        plt.plot(history.history['val_accuracy'], label = 'validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.ylim([0.5, 1])
        plt.title("Training Accuracy vs Validation Accuracy")
        plt.show()
    
    def plot_dice():
        plt.plot(history.history['dice_coef'],'seagreen', label='dice_coef')
        plt.plot(history.history['val_dice_coef'], label='val_dice_coef')
        plt.xlabel("Epoch")
        plt.ylabel("Dice Coefficient")
        plt.legend(loc='lower right')
        plt.ylim([0.5, 1])
        plt.title("Training Dice Coefficient vs Validation Dice Coefficient")
        plt.show()
    
    def plot_loss():
        plt.plot(history.history['loss'],'seagreen', label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='lower right')
        plt.title("Training Loss vs Validation Loss")
        plt.show()
    
    plot_accuracy()
    plot_dice()
    plot_loss()
    
    "Show a complete model summary"
    model.summary()