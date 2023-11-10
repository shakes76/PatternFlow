#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:14:07 2021

@author: hongweili
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import glob as gb
from sklearn.model_selection import train_test_split
from unet_model import *
import tensorflow.keras.backend as backend

print('TensorFlow version:', tf.__version__)

# if GPU is avaliable, use GPU
physical_devices = tf.config.list_physical_devices('GPU')
print(len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    
class DATA_PREPROCESS:
    def __init__(self):
        # resized image size
        self.height = 256
        self.width = 192

        # default image path
        self.image_path =  "ISIC/ISIC2018_Task1-2_Training_Input_x2/*.jpg"
        self.mask_path = "ISIC/ISIC2018_Task1_Training_GroundTruth_x2/*.png"
        
    '''
        load images and split raw data into training set, validating set and test set 
    '''
    def load_data(self):
        # load images
        images = sorted(gb.glob(self.image_path))
        masks = sorted(gb.glob(self.mask_path))
        print(str(len(images)),  ' raw images loaded')
        print(str(len(masks)),  ' raw masks loaded')
        
        # split data into training set, validating set and testing set with the percentage of 60%, 20%, 20%
        X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size = 0.4, random_state = 1)
        X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size = 0.5, random_state = 1)
        print('Number of training sets: ', str(len(X_train)))
        print('Number of validating sets: ', str(len(X_validate)))
        print('Number of testing sets: ', str(len(X_test)))
        
        #prepare the tf data for UNet model
        self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.validate_dataset = tf.data.Dataset.from_tensor_slices((X_validate, y_validate))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # shuffle the all datasets
        self.train_dataset.shuffle(len(X_train))
        self.validate_dataset.shuffle(len(X_validate))
        self.test_dataset.shuffle(len(X_test))
        self.num_test = len(X_test)
        
    '''
        resize image, normalize image and reshape image
    '''
    def rescale_image(self, image_path, mask_path):
        # resize image 
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels = 1)
        image = tf.image.resize(image, [self.height, self.width])
        # normalize image 
        image = tf.cast(image, tf.float32) / 255.0
        # reshape imahe 
        image = tf.reshape(image, (self.height, self.width, 1))

        # resize mask                            
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels = 1)
        mask = tf.image.resize(mask, [self.height, self.width])
        # normalize mask 
        mask = tf.cast(mask, tf.float32) / 255.0
        # reshape mask
        mask = tf.reshape(mask, (self.height, self.width, 1))
        return image, mask
    
    
    '''
        normalized and resize & reshape all datasets into predefined size (i.e. 256 * 192 * 1)
    '''
    def preprocess(self):
        self.train_dataset = self.train_dataset.map(self.rescale_image)
        self.validate_dataset = self.validate_dataset.map(self.rescale_image)
        self.test_dataset = self.test_dataset.map(self.rescale_image)
    

    '''
        display the result of rescaled image
    '''
    def display_preprocessed_image(self, num_sets):
        for image, mask in self.train_dataset.take(num_sets):
            plt.figure(figsize=(12,12))
            plt.subplot(1,2,1)
            plt.imshow(tf.squeeze(image), cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(tf.squeeze(mask), cmap='gray')
        plt.show()

#######################################################################################
#                      SUPPORT FUNCTIONS OR CLASSES IN TRAINING                       #
#######################################################################################

'''
    Calculates the Dice coefficient of two provided
    tensors.

    Author: Hadrien Mary
    Retrieved from: https://github.com/keras-team/keras/issues/3611
'''
def dice_coef(y_true, y_pred, smooth=0.00001):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)


"""
Predicts a mask based on image provided, and displays
the predicted mask alongside the actual segmentation
mask and original image.
"""
def show_pred_mask(ds, num = 1):
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis, ...])[0]
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(1,3,1)
        # plot raw image 
        ax1.imshow(tf.squeeze(image), cmap='gray')
        ax1.set_title('Raw Grayscale Image')
        ax2 = fig.add_subplot(1,3,2)
        # plot raw mask
        ax2.imshow(tf.squeeze(mask), cmap='gray')
        ax2.set_title('Raw Mask')
        ax3 = fig.add_subplot(1,3,3)
        # plot predicted mask
        ax3.imshow(tf.squeeze(pred_mask), cmap='gray')
        ax3.set_title('Predicted Mask')
#         print(dice_coef(mask, pred_mask))
    plt.show()


"""
    Display the predicted mask at the end of each epoch
"""
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        show_pred_mask(validate)

"""
    Plot the train history of UNet model, including accuracy, DSC and loss 
"""
def plot_train_history():
    # Plot Accuracy Trend
    fig = plt.figure(figsize=(8,27))
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(history.history['accuracy'], 'blue', label='train')
    # plot validating dataset accuracy
    ax1.plot(history.history['val_accuracy'], 'green', label = 'validation')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc='lower right')
    ax1.set_title("Training Accuracy vs Validation Accuracy")
    
    # Plot DSC Trend
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(history.history['dice_coef'], 'purple', label='train')
    # plot validating dataset accuracy
    ax2.plot(history.history['val_dice_coef'], 'red', label = 'validation')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc='lower right')
    ax2.set_title("Training Dice Coefficient vs Validation Dice Coefficient")
    
    # Plot Loss Trend
    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(history.history['loss'],'Navy', label='train')
    # plot validation loss
    ax3.plot(history.history['val_loss'],'Orange', label='validation')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend(loc='lower right')
    ax3.set_title("Training Loss vs Validation Loss")
    
    plt.show()


"""
    Calculate the average test DSC using trained model
"""
def cal_test_DSC(dataset, num_test):
    sum = 0
    # num_test = data.num_test
    for image, mask in dataset.take(num_test):

        mask_pred = model.predict(image[tf.newaxis, ...])[0]

        dsc = dice_coef(mask, mask_pred)
        sum += dsc
    #     print(dsc)
    sum /= num_test
    return sum







###########################################################################################
#                                   MODEL DRIVER                                          #
###########################################################################################

print('=====> Begin to preprocess the raw images and masks')
data = DATA_PREPROCESS()
data.load_data()
data.preprocess()
data.display_preprocessed_image(1)
train = data.train_dataset
validate = data.validate_dataset 
test = data.test_dataset 
num_test = data.num_test
print('=====> Finish preprocessing the raw images and masks\n')
print('=====> Begin to train UNet Model')
model = Improved_UNet((data.height, data.width, 1))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', dice_coef])
history = model.fit(train.batch(12), epochs = 200, validation_data = validate.batch(12), callbacks = [DisplayCallback()])
plot_train_history()
print('=====> Finish model trainig and history displaying\n')
print('=====> Use trained model to perform segmentation')
# Show model predicted masks.
show_pred_mask(test, 4)
dsc = cal_test_DSC(test, num_test)
tf.print('Average test DSC of this model: ', dsc)
print('=====> Finish trained model segmentation')




    
