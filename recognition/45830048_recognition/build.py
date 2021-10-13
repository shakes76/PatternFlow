"""
Driver Script
"""

import tensorflow as tf
from model import *
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

############LOAD CELEB-A####################
diretory      = "recognition/45830048_recognition/BCN_2020_Challenge/"
no_train        = 15
no_test         = 5
images       = np.sort(os.listdir(diretory))
## name of the jpg files for training set
train_images = images[:no_train]
## name of the jpg files for the testing data
test_images  = images[no_train:no_train + no_test]
image_res     = (640, 480, 3)

def images_to_array(nm_imgs_train):
    images_array = []
    for null, image_no in enumerate(nm_imgs_train):
        image = load_img(diretory + "/" + image_no,
                         target_size=(64, 64))
        image = img_to_array(image)/255
        images_array.append(image)
    images_array = np.array(images_array)
    return images_array

train = images_to_array(train_images)
print("X_train.shape = {}".format(train.shape))

test  = images_to_array(test_images)
print("X_test.shape = {}".format(test.shape))

fig = plt.figure(figsize=(30,10))
nplot = 7
for count in range(1,nplot):
    ax = fig.add_subplot(1,nplot,count)
    ax.imshow(train[count])
plt.show()