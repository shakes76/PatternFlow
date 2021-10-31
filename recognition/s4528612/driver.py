
import re

import tensorflow as tf
import random, os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras as keras
from tensorflow.keras import layers

import os
from model import Perceiver
import random, os
import numpy as np

IMAGE_DIR = '../input/knee-data/AKOA_Analysis/'
EPOCHS = 10

def data_processing(directory, train,validation):
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
    
    for index, patient in enumerate(patient_knee_collection.items()):
        for i in range(len(patient[1])):
            if (index/len(list(patient_knee_collection.items()))) < train:
                x_train.append(img_to_array(load_img(directory + patient[1][i], target_size=IMG_SIZE, color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_train.append(0)
                else:
                    left += 1
                    y_train.append(1)
                    
            if (index/len(list(patient_knee_collection.items()))) < (train+validation):
                x_test.append(img_to_array(load_img(directory + patient[1][i], target_size=IMG_SIZE, color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_test.append(0)
                else:
                    left += 1
                    y_test.append(1)
            else:
                x_val.append(img_to_array(load_img(directory + patient[1][i], target_size=IMG_SIZE, color_mode="grayscale")))
                if re.search('right', patient[1][i].replace("_",""), re.IGNORECASE):
                    right += 1
                    y_test.append(0)
                else:
                    left += 1
                    y_test.append(1)                
                           
    x_train = np.array(x_train)
    x_train /= 255.0
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_test /= 255.0
    y_test = np.array(y_test)
    x_val = np.array(x_val)
    x_val /= 255.0
    y_val = np.array(y_val)
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
    X_train, y_train, X_val, y_val, X_test, y_test = data_processing(IMAGE_DIR, TRAIN_SPLIT,0.04)
    perceiver = Perceiver()
    
    X_train = X_train[0:len(X_train) // 32 * 32]
    y_train = y_train[0:len(y_train) // 32 * 32]
    X_val =  X_val[0:len(X_val) // 32 * 32]
    y_val = y_val[0:len(y_val) // 32 * 32]
    X_test = X_test[0:len(X_test) // 32 * 32]
    y_test = y_test[0:len(y_test) // 32 * 32]
    
    history = train_perceiver(perceiver,  X_train, y_train,X_val, y_val,X_test, y_test,epochs=EPOCHS)

if __name__ == "__main__":
    main()