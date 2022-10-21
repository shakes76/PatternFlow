

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import to_categorical, Sequence
import project as p
import model as m
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import random

def encode_y(y):

    """
    Categorical encoding of grond truth values used in training
    
    @param y -- ground truth batch
    
    """
    
    y = tf.keras.utils.to_categorical(y, num_classes=8)
    return y

# A class that can be parsed in to the fit model parameter
# to have its functions called natively by fit model. 
#code ref
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

class ISICs2018Sequence(Sequence):
    
    """
    A class defining a Sequence to be used iteratively for each batch and epoch. 
    
    @param Sequence -- A keras Sequence
    
    """
    
    def __init__(self, x, y, batchsize):
    
        """
        Initialises the Sequence
        
        @param x -- the subject images in the batch
        @param y -- the ground truth images in the batch
        @param batchsize -- the number of images in the batch
        
        """
        
        self.x = x
        self.y = y
        self.batchsize = batchsize
        
    def __len__(self):
        
        """
     
        Returns the length of this Sequence object

        """
        
        return math.ceil(len(self.x) / self.batchsize)
    
    def __getitem__(self, id):
        
        """
        
        Gets the subject image and ground truth image associated with the current batch for the current image id that the model is processing. 
        Gurantees training on each sample per epoch. 
        
        @param id - The current id of the subject image being used in the model. 
        
        """
        
        x_names = self.x[id * self.batchsize:(id + 1) * self.batchsize]
        y_names = self.y[id * self.batchsize:(id + 1) * self.batchsize]
        
        x_batch = list()
        y_batch = list()
        
        for name in x_names :
            
            file_name = name[:len(name) - 4]
            
            train_image = np.asarray(Image.open("ISIC2018_Task1-2_Training_Input_x2/" + file_name + ".jpg").resize((256, 192))) / 255.0
        
            x_batch.append(train_image)
        
            ground_image = np.asarray(Image.open("ISIC2018_Task1_Training_GroundTruth_x2/" + file_name + "_segmentation" + ".png").resize((256, 192))) / 255.0 
           
            y_batch.append(ground_image)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        
        y_batch = encode_y(y_batch)
        
        return x_batch, y_batch

def load_data(directory, seed) :
    
    """
    
    Loads the data using a random seed to shuffle post load
    
    @param seed - the random seed to be used to shuffle the data
    
    """
    
    loaded = list()
    
    for file in os.listdir(directory) :
    
        if file != "ATTRIBUTION.txt" and file != "LICENSE.txt" :
            
            loaded.append(file)

    random.seed(seed)
    random.shuffle(loaded)

    return loaded

def plot_result(history):

    """ 
    Plots the number of epochs vs the average dice similarity
    
    """

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    
    plt.plot(history.history['dice'], label='dice')
    plt.plot(history.history['val_dice'], label = 'val_dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice / Accuracy Coefficient')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    
    plt.show()

def show_images(test, model) :
    
    """
    
    Shows the images from the predicted model result next to the test result
    for the associated ground truth images. 
    
    @param test - the test Sequence 
    @param model - the model generated
    
    """
    
    testlength = len(test)
    rand = random.randint(0, testlength)
    x, y = test.__getitem__(rand)
    prediction = model.predict(x)
    plt.figure(figsize=(10, 10))
    
    for i in range(8):
           
        plt.subplot(9, 3, i*3+1)
        plt.imshow(x[i])
        plt.axis('off')
        
        if i == 0 :
            plt.title("Original Image", size=8)     
        
        plt.subplot(9, 3, i*3+2)
        plt.imshow(tf.argmax(prediction[i], axis=2))
        plt.axis('off')
        
        if i == 0 :
            plt.title("Model Output", size=8)
        
        plt.subplot(9, 3, i*3+3)
        plt.imshow(tf.argmax(y[i], axis=2))
        plt.axis('off')
        
        if i == 0 :
            plt.title("What it should be", size=8)
    
    plt.show()

if __name__ == "__main__":

    """
    
    Loads the sets, segements and shuffles the sets into training, validation and test, then runs the model.
    
    The directory for the data sets must be set in the associated strings.
    training_directory and ground_directory.

    """

    training_directory = "D:/3710sets/2018/ISIC2018_Task1-2_Training_Input_x2"
    ground_directory = "D:/3710sets/2018/ISIC2018_Task1_Training_GroundTruth_x2"

    seed = random.random()
    train = load_data(training_directory, seed)
    ground = load_data(ground_directory, seed)

    validation_prop = 0.2
    test_prop = validation_prop
    batch_size = 8

    train_images, test_images, ground_images, ground_test = train_test_split(train, ground, test_size=validation_prop, random_state=50)

    train_images, val_images, ground_images, ground_val = train_test_split(train_images, ground_images, test_size=test_prop, random_state=50)

    train = ISICs2018Sequence(train_images, ground_images, batch_size)
    val = ISICs2018Sequence(val_images, ground_val, batch_size)
    test = ISICs2018Sequence(test_images, ground_test, batch_size)
    
    model = p.UNET()
    
    history = model.fit(train,
          validation_data=val,
          epochs=30, verbose=1, workers=4)
    
    model.evaluate(test)
    
    plot_result(history)
    
    show_images(test, model)