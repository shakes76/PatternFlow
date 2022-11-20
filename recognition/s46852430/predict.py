# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:08:16 2022

@author: eudre
"""
import matplotlib.pyplot as plt
import dataset
import glob
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras.optimizers import SGD
import model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping



train_path ='C:/Users/eudre/test/ISIC-2017_Training_Data/*.jpg'
mask_path ='C:/Users/eudre/test/ISIC-2017_Training_Part1_GroundTruth/*.png'

def dice_coef(y_true, y_pred, epsilon=0.00001):

    axis = (0,1,2,3)
    dice_numerator = 2. * backend.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = backend.sum(y_true*y_true, axis=axis) + backend.sum(y_pred*y_pred, axis=axis) + epsilon
    return backend.mean((dice_numerator)/(dice_denominator))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


image = sorted(glob.glob(train_path))
groundtruth = sorted(glob.glob(mask_path))

(train_set), (valid_set), (test_set) = dataset.spilt_data(train_path, mask_path)


training_set = train_set.map(dataset.load_data)
validation_set=valid_set.map(dataset.load_data)
test_set=test_set.map(dataset.load_data)

batch_size = 32

num_epoch = 5

learning_rate = 0.1
decay_rate = learning_rate / num_epoch
momentum = 0.9

opt = SGD(learning_rate = learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

callback = [
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, min_lr=1e-7, verbose=2),
    EarlyStopping(monitor='loss', patience = 1, restore_best_weights=False)
]


model = model.modified_UNET((256,256,3))
model.compile(optimizer= opt, loss=dice_coef_loss, metrics=[dice_coef])
#Train the model
model_history=model.fit(training_set.batch(batch_size), validation_data=validation_set.batch(batch_size), epochs=num_epoch, callbacks = callback)
#Evaluate
result=model.evaluate(test_set.batch(batch_size), verbose=1)


input_size=(0, 0, 0)
number_output_classes=0
#Taking only one is enough
for image, label in training_set.take(1):
    input_size=image.numpy().shape
    number_output_classes=label.numpy().shape[2]
plt.figure(figsize=(25, 25))
plt.subplot(1, 4, 1)
plt.imshow(image.numpy())
plt.axis('off')
plt.subplot(1, 4, 2)
if (number_output_classes>1):
    plt.imshow(tf.argmax(label.numpy(),axis=2))
else:
    plt.imshow(label.numpy())
plt.axis('off')

def display_images(image, groundtruth, prediction, number):
    plt.figure(figsize=(20, 20))
    colors = ['black', 'grey', 'white']
    for i in range(number):
        plt.subplot(4, 3, 3*i+1)
        plt.imshow(image[i])
        title = plt.title('Origin Image')
        plt.setp(title, color=colors[0])
        plt.axis('off')
        plt.subplot(4, 3, 3*i+2)
        if (number_output_classes > 1):
            plt.imshow(tf.argmax(groundtruth[i], axis=2))
        else:
            plt.imshow(groundtruth[i])
        title = plt.title('Ground Truth Segmentation')
        plt.setp(title, color=colors[1])
        plt.axis('off')
        plt.subplot(4, 3, 3*i+3)
        if (number_output_classes > 1):
            plt.imshow(tf.argmax(prediction[i], axis=2))
        else:
            plt.imshow(prediction[i] > 0.5)
        title = plt.title('Prediction Segmentation')
        plt.setp(title, color=colors[2])
        plt.axis('off')
    plt.show()


batch_image, batch_label = next(iter(test_set.batch(3)))
prediction = model.predict(batch_image)
display_images(batch_image, batch_label, prediction, 3)
