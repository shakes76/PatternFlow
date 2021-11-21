"""
Author: Humphrey Munn
Student Number: 45839724
COMP3710 Sem2, 2021.

Driver script for running knee laterality perceiver model. 
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
tf.compat.v1.enable_eager_execution()
from perceiver_model import Perceiver

EPOCHS = 3
BATCH_SIZE = 8
DATA_DIR = r"C:\Users\hmunn\OneDrive\Desktop\COMP3710\Project\Data\AKOA_Analysis\\"
TEST_SPLIT = 0.4

''' Processes AKOA data and saves train and test sets. '''
def save_data():
    # LOAD IN DATA. Organize by patient, to prevent data leakage. 
    file_paths = [DATA_DIR + x for x in os.listdir(DATA_DIR)]
    new_patient_ids = {} # key: e.g. OAI9014797_3_L, value: new id
    data = {} # key: unique patient id (created), value: ([xdata], [labels [0 for left, 1 for right]])
    totals = 0
    right = 0
    for file in file_paths:
        # 4 ways filenames specify right knees
        is_right = "RIGHT" in file or "Right" in file or "right" in file or "R_I_G_H_T" in file
        right += 1 if is_right else 0
        totals += 1
        info = file.split("_BaseLine_")
        # get unique patient id based on first part string, second number and left/right
        patient_id = info[0] + "_" + info[1].split("de3")[0].split("_")[0] + "_" + ("L" if not is_right else "R")
        if patient_id not in new_patient_ids:
            new_patient_ids[patient_id] = len(new_patient_ids)
        new_id = new_patient_ids[patient_id]
        # load in image and normalize, and assign label
        img = np.asarray(Image.open(file).convert("L"))
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        label = 1 if is_right else 0
        # add data to dictionary
        if new_id in data:
            data[new_id][0].append(img)
            data[new_id][1].append(label)
        else:
            data[new_id] = ([img], [label])

    # SPLIT DATA. Get train/test split based on patients. 
    num_patients = len(data)
    patient_ids = list(range(0, num_patients))
    test_patients = random.sample(patient_ids, int(num_patients*TEST_SPLIT))
    train_patients = [x for x in patient_ids if x not in test_patients]

    xtrain, xtest, ytrain, ytest = [], [], [], []
    for pid in patient_ids:
        # add train/test data based on the indicies from above
        for idx in range(len(data[pid][0])):
            if pid in train_patients:
                xtrain.append(data[pid][0][idx])
                ytrain.append(data[pid][1][idx])
            else:
                xtest.append(data[pid][0][idx])
                ytest.append(data[pid][1][idx])
    print(len(xtrain), len(xtest), len(ytrain), len(ytest))
    del data

    # SHUFFLE DATA AND SAVE. 
    indices_train = list(range(0, len(xtrain)))
    indices_test = list(range(0, len(xtest)))
    random.shuffle(indices_train)
    random.shuffle(indices_test)
    xtrain = np.array(xtrain)
    xtrain = xtrain[indices_train]
    np.save("xtrain", xtrain)
    del xtrain
    xtest = np.array(xtest)
    xtest = xtest[indices_test]
    np.save("xtest", xtest)
    del xtest
    ytrain = np.array(ytrain)
    ytrain = ytrain[indices_train]
    np.save("ytrain", ytrain)
    del ytrain
    ytest = np.array(ytest)
    ytest = ytest[indices_test]
    np.save("ytest", ytest)
    del ytest

''' Ensures data is divisible/within range of batch size, if not removes data off the end. '''
def create_batches_from_data(xdata, ydata, batches):
    xdata_new = xdata[:int(len(xdata) / batches) * batches]
    ydata_new = ydata[:int(len(ydata) / batches) * batches]
    return (xdata_new, ydata_new)

''' Returns the learning rate given the epoch (refer to Perceiver paper). '''
def learning_rate_decay(epoch):
    lr = 0.001
    decay_epochs = [84, 102, 114]
    for ep in decay_epochs:
        if epoch >= ep:
            lr /= 10
    return (lr)

''' Plot accuracy and loss for model history. '''
def plot_history(history):
    # Plot learning history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss (Binary Cross-Entropy)')
    plt.xlabel('Epochs')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    # Save data, then load back in (comment out if already saved)
    # save_data()
    xtrain = np.load(r"xtrain.npy")
    #xtrain = xtrain[0:2000]
    xtrain = np.reshape(xtrain, (*xtrain.shape, 1))
    ytrain = np.load(r"ytrain.npy")
    #ytrain = ytrain[0:2000]

    # Change index ranges to choose validation set that is not in training set (or use test set)
    xval = np.load(r"xtrain.npy")
    xval = xval[250:700]
    xval = np.reshape(xval, (*xval.shape, 1))
    yval = np.load(r"ytrain.npy")
    yval = yval[250:700]

    # Compile and run perceiver model
    perceiver = Perceiver()
    learning_rate_fnc = tf.keras.callbacks.LearningRateScheduler(learning_rate_decay)

    # Compile the model
    perceiver.compile(
        optimizer = tfa.optimizers.LAMB(learning_rate=0.001, weight_decay_rate = 0.0002),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = tf.keras.metrics.BinaryAccuracy(name="accuracy")
    )
    
    history = perceiver.train_model(xtrain, ytrain, xval, yval, epochs = EPOCHS, batches = BATCH_SIZE, lr_func=learning_rate_fnc)

    # perceiver.save("./perceiver_model")

    del xtrain
    del xval
    del ytrain
    del yval

    plot_history(history)

    # evaluate model
    xtest, ytest = np.load(r"xtest.npy"), np.load(r"ytest.npy")
    xtest = np.reshape(xtest, (*xtest.shape, 1))
    xtest, ytest = create_batches_from_data(xtest, ytest, BATCH_SIZE)

    '''
    # Uncomment to predict 8 random images.
    import random
    choices = [random.choice(list(range(len(xtest)))) for i in range(8)]
    imgs = np.array([xtest[j] for j in choices])
    imgs = imgs[:int(len(imgs) / 8) * 8]
    print(imgs.shape)
    predictions = perceiver.predict(imgs)

    for idx, p in enumerate(predictions):
        plt.imshow(imgs[idx])
        print("prediction:", p, "actual:", ytest[choices[idx]])
        plt.show()
    '''

    # Test accuracy
    _, acc = perceiver.evaluate(xtest, ytest, batch_size = BATCH_SIZE)
    print(f"Accuracy on test set:{round(acc * 100, 4)}%")
