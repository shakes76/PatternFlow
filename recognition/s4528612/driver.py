import re

import tensorflow as tf
import random, os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import os

import random, os
import numpy as np
from model import Perceiver

"""
Main Function running the training of the model and the visualisations of the predictions.
"""
def main():
    # File Path Containing Knee Images
    IMAGE_DIR = '../input/knee-data/AKOA_Analysis/'
    # Epochs for training the perceiver
    EPOCHS = 10
    # Produce the sets for training, validation and testing.
    X_train, y_train, X_val, y_val, X_test, y_test = process_dataset(IMAGE_DIR, 0.8,0.04)
    # Perceiver Instantiation
    perceiver = Perceiver()
    # Ensure training set doesn not exceed batch size to avoid fourier encoder error
    X_train = X_train[0:len(X_train) // 32 * 32]
    y_train = y_train[0:len(y_train) // 32 * 32]
    X_val =  X_val[0:len(X_val) // 32 * 32]
    y_val = y_val[0:len(y_val) // 32 * 32]
    X_test = X_test[0:len(X_test) // 32 * 32]
    y_test = y_test[0:len(y_test) // 32 * 32]
    # Run Perceiver Training Function, returning accuracy data
    history = train_perceiver(perceiver,  X_train, y_train,X_val, y_val,X_test, y_test,epochs=EPOCHS)
    # Plot Accuracies
    accuracy = history.history['acc']
    validation_accuracy = history.history['val_acc']
    plt.figure()
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy of Perceiver on OAI AKOA Dataset Laterality classification')
    plt.show()
    # Reshape Test Set to Visualize images
    test_images = X_test[:32].reshape((32, 73, 64, 1))
    test_labels = y_test[:32].flatten()
    # Return Predictions
    predictions = tf.where(perceiver.predict_on_batch(test_images).flatten() < 0.5, 0, 1).numpy()
    # Plot Knees and predictions of the test set
    for i in range(32):
        plt.imshow(X_test[i], cmap="gray")
        # Classifications of the Knee
        laterality = {0: "right", 1: "left"} 
        if predictions[i] < 0.5:
            plt.title("Predicted Right," "Actually: " + laterality[test_labels[i]])
        else:
            plt.title("Predicted Left," "Actually: " + laterality[test_labels[i]])

        plt.show() # This Line may only produce one plot at a time on a normal python environment. On Kaggle it shows all images. 

"""
Shuffles the input dictionary
Params:
    dictionary: Dictionary to be shuffled
Returns: A shuffled version of the input directory
"""
def shuffle_dictionary(dictionary):
    items = list(dictionary.items())
    random.shuffle(items)
    return dict(items)

"""
Given the image directory, the train and validation split, returns the training, testing and validation sets.
Params:
    directory: Directory containing the knee images
    train: The training split as a decimal less than 1.
    validation: The validation split as a decimal less than 1.
Returns: The training, testing and validation sets.
"""
def process_dataset(directory, train,validation):
    left = 0
    right = 0
    labels = []
    patient_knee_collection = dict({})

    # Parse all knee images and sort based on their patient_id to ensure that the same patient doesn't end up in both the test and training set. 
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            patient_id = filename.split('_')[0]
            if patient_id in patient_knee_collection:
                patient_knee_collection[patient_id].append(filename) 
            else:
                patient_knee_collection[patient_id] = [filename]
    # Shuffle the Dictionary
    patient_knee_collection = shuffle_dictionary(patient_knee_collection)
    # Initializing sets
    x_train = []
    y_train = [] 
    x_test = [] 
    y_test = []
    x_val = [] 
    y_val = []
    # Parse all knees and sort and add to relevent sets
    for index, patient in enumerate(patient_knee_collection.items()):
        for i in range(len(patient[1])):
            # Training Set
            if (index/len(list(patient_knee_collection.items()))) < train:
                x_train.append(img_to_array(load_img(directory + patient[1][i], target_size=(73,64), color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_train.append(0)
                else:
                    left += 1
                    y_train.append(1)
            #  Validation Set
            if (index/len(list(patient_knee_collection.items()))) < (train+validation):
                x_test.append(img_to_array(load_img(directory + patient[1][i], target_size=(73,64), color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_test.append(0)
                else:
                    left += 1
                    y_test.append(1)
            # Test Set
            else:
                x_val.append(img_to_array(load_img(directory + patient[1][i], target_size=(73,64), color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_val.append(0)
                else:
                    left += 1
                    y_val.append(1)

                    
                    
    # Convert Sets to numpy arrays and normalize the image arrays    
    x_train = np.array(x_train)
    x_train /= 255.0
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_test /= 255.0
    y_test = np.array(y_test)
    x_val = np.array(x_val)
    x_val /= 255.0
    y_val = np.array(y_val)
    
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    print(x_test.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test


"""
Given the image directory, the train and validation split, returns the training, testing and validation sets.
Params:
    directory: Directory containing the knee images
    train: The training split as a decimal less than 1.
    validation: The validation split as a decimal less than 1.
Returns: The training, testing and validation sets.
"""

def train_perceiver(perceiver,X_train, y_train,X_val, y_val,X_test, y_test,epochs):
    # Compile Perceiver with Lamb Optimizer 
    perceiver.compile(optimizer=tfa.optimizers.LAMB(learning_rate=0.001, weight_decay_rate=0.0001,), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")])
    history = perceiver.fit(
        X_train, y_train,
        epochs= epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        validation_batch_size=32
    )
    # Calculate Test Accuracy (Final Result)
    _, accuracy = perceiver.evaluate(X_test, y_test)
    print("Test accuracy:" + str(accuracy))
    return history


if __name__ == "__main__":
    main()