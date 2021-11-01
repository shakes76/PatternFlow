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
def main():


    IMAGE_DIR = '../input/knee-data/AKOA_Analysis/'
    EPOCHS = 10

    X_train, y_train, X_val, y_val, X_test, y_test = process_dataset(IMAGE_DIR, 0.8,0.04)
    perceiver = Perceiver()
    
    # Ensure training set doesn not exceed batch size to avoid fourier encoder error
    X_train = X_train[0:len(X_train) // 32 * 32]
    y_train = y_train[0:len(y_train) // 32 * 32]
    X_val =  X_val[0:len(X_val) // 32 * 32]
    y_val = y_val[0:len(y_val) // 32 * 32]
    X_test = X_test[0:len(X_test) // 32 * 32]
    y_test = y_test[0:len(y_test) // 32 * 32]
    
    history = train_perceiver(perceiver,  X_train, y_train,X_val, y_val,X_test, y_test,epochs=EPOCHS)

    
    
    accuracy = history.history['acc']
    validation_accuracy = history.history['val_acc']
    plt.figure()

    #Plot Accuracies
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy of Perceiver on OAI AKOA Dataset Laterality classification')
    plt.show()
    
    test_images = X_test[:32].reshape((32, 73, 64, 1))
    test_labels = y_test[:32].flatten()
    
    predictions = tf.where(perceiver.predict_on_batch(test_images).flatten() < 0.5, 0, 1).numpy()
    
    for i in range(32):
        plt.imshow(X_test[i], cmap="gray")
        laterality = {0: "right", 1: "left"} 
        if predictions[i] < 0.5:
            plt.title("Predicted Right," "Actually: " + laterality[test_labels[i]])
        else:
            plt.title("Predicted Left," "Actually: " + laterality[test_labels[i]])

        plt.show() # This Line may only produce one plot at a time on a normal python environment. On Kaggle it shows all images. 


def shuffle_dictionary(dictionary):
    items = list(dictionary.items())
    random.shuffle(items)
    return dict(items)


def process_dataset(directory, train,validation):
    left = 0
    right = 0
    labels = []


    patient_knee_collection = dict({})

    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            patient_id = filename.split('_')[0]
            if patient_id in patient_knee_collection:
                patient_knee_collection[patient_id].append(filename) 
            else:
                patient_knee_collection[patient_id] = [filename]

    patient_knee_collection = shuffle_dictionary(patient_knee_collection)

    x_train = []
    y_train = [] 
    x_test = [] 
    y_test = []
    x_val = [] 
    y_val = []
    #j = 0
    
    for index, patient in enumerate(patient_knee_collection.items()):
        for i in range(len(patient[1])):

            if (index/len(list(patient_knee_collection.items()))) < train:
                x_train.append(img_to_array(load_img(directory + patient[1][i], target_size=(73,64), color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_train.append(0)
                else:
                    left += 1
                    y_train.append(1)

            if (index/len(list(patient_knee_collection.items()))) < (train+validation):
                x_test.append(img_to_array(load_img(directory + patient[1][i], target_size=(73,64), color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_test.append(0)
                else:
                    left += 1
                    y_test.append(1)
            else:
                x_val.append(img_to_array(load_img(directory + patient[1][i], target_size=(73,64), color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_val.append(0)
                else:
                    left += 1
                    y_val.append(1)

                    
                    
        
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




def train_perceiver(perceiver,X_train, y_train,X_val, y_val,X_test, y_test,epochs):

    perceiver.compile(optimizer=tfa.optimizers.LAMB(learning_rate=0.001, weight_decay_rate=0.0001,), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")])
    history = perceiver.fit(
        X_train, y_train,
        epochs= epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        validation_batch_size=32
    )
    _, accuracy = perceiver.evaluate(X_test, y_test)
    print("Test accuracy:" + str(accuracy))
    return history



def main():

    IMAGE_DIR = '../input/knee-data/AKOA_Analysis/'
    EPOCHS = 10
    X_train, y_train, X_val, y_val, X_test, y_test = process_dataset(IMAGE_DIR, 0.8,0.04)
    perceiver = Perceiver()
    
    X_train = X_train[0:len(X_train) // 32 * 32]
    y_train = y_train[0:len(y_train) // 32 * 32]
    X_val =  X_val[0:len(X_val) // 32 * 32]
    y_val = y_val[0:len(y_val) // 32 * 32]
    X_test = X_test[0:len(X_test) // 32 * 32]
    y_test = y_test[0:len(y_test) // 32 * 32]
    
    history = train_perceiver(perceiver,  X_train, y_train,X_val, y_val,X_test, y_test,epochs=EPOCHS)

    
    
    accuracy = history.history['acc']
    validation_accuracy = history.history['val_acc']
    plt.figure()

    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy of Perceiver on OAI AKOA Dataset Laterality classification')
    plt.show()
    
    test_images = X_test[:32].reshape((32, 73, 64, 1))
    test_labels = y_test[:32].flatten()
    
    predictions = tf.where(perceiver.predict_on_batch(test_images).flatten() < 0.5, 0, 1).numpy()
    
    for i in range(32):
        plt.imshow(X_test[i], cmap="gray")
        laterality = {0: "right", 1: "left"} 
        if predictions[i] < 0.5:
            plt.title("Predicted Right," "Actually: " + laterality[test_labels[i]])
        else:
            plt.title("Predicted Left," "Actually: " + laterality[test_labels[i]])

        plt.show() # This Line may only produce one plot at a time on a normal python environment. On Kaggle it shows all images. 
if __name__ == "__main__":
    main()