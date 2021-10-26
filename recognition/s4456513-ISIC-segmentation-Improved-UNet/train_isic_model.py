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
            
        


def main():
    preprocessd_data = DATA_PREPROCESS()
    preprocessd_data.load_data()
    preprocessd_data.preprocess()
    preprocessd_data.display_preprocessed_image(1)
    
if __name__ == '__main__':
    main()
