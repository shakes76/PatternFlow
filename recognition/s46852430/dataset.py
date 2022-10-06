# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:57:39 2022

@author: eudre
"""

"""
Created on Mon Oct  3 17:29:06 2022

@author: eudre
"""
#importing the libraries
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path ='C:\\Users\\eudre\\test\\AD_NC\\train'
test_path ='C:\\Users\\eudre\\test\\AD_NC\\test'

class_names = ['AD', 'NC']

# apply glob module to retrieve files/pathnames  

x_train=[]
x_test=[]
for folder in os.listdir(train_path):
    sub_path=train_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224,224))

        x_train.append(img_arr)
    
for folder in os.listdir(test_path):
    sub_path=test_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224,224))

        x_test.append(img_arr)
    
train_x=np.array(x_train)
test_x=np.array(x_test)

train_x=train_x/255.0
test_x=test_x/255.0

train_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'sparse')


train_y=training_set.classes
test_y=test_set.classes




