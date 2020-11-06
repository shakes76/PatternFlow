#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import cv2
import random
import zipfile
import wget
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print(tf.__version__)


# Need to convert arrays to tensors in future code
def conv_tensor(arg):
    """ Function to convert arrays to tensors.
       Parameters:
            arg: the numpy array/list/scalar to be converted to a tensor.
       Return:
            arg: the now converted Tensorflow tensor."""
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


# Want to try and download the dataset from URL and unzip
url = "https://cloudstor.aarnet.edu.au/sender/download.php?token=d82346d9-f3ca-48bf-825f-327a622bfaca&files_ids=9881639"

wget.download(url)

with zipfile.ZipFile("AKOA_Analysis.zip", "r") as zip_ref:
    zip_ref.extractall()
# file now in directory called AKOA_Analysis, that contains all the images


data_dir = "AKOA_Analysis"

data_dir = pathlib.Path(data_dir)

total_images = list(data_dir.glob('*.png'))
print('total images', len(total_images))

# Not sure how much of beneath is useful
left_images = list(data_dir.glob('*left*.png'))
left_underscored = list(data_dir.glob('*L_E_F_T*.png'))
left_images = left_images + left_underscored
print('left images', len(left_images))

right_images = list(data_dir.glob('*right*.png'))
right_underscored = list(data_dir.glob('*R_I_G_H_T*.png'))
right_images = right_images + right_underscored
print('right count', len(right_images))

print('sum ', len(right_images)+len(left_images))
# above stuff may not be useful

# Randomising the collection of images
print('total images first', total_images[1])
random.seed(87)
random.shuffle(total_images)
# checking to see if the total_images have shuffled
print('total images first post shuffle', total_images[1])


# initialising the labels array 
total_images_labels = np.array([None]*len(total_images))
left = ["left", "l_e_f_t"]
right = ["right", "r_i_g_h_t"]

# labelling the images based on contents of path
for i in range(len(total_images)):
    path = total_images[i]
    # make path names all lowercase
    path = str(path).lower()
    
    # check laterality and give label of 0 (left) or 1(right) for that index of labels array
    if any(x in path for x in left):
        total_images_labels[i] = 0  # left
    elif any(x in path for x in right):
        total_images_labels[i] = 1  # right
    else:
        print("Unclassified Images: ", total_images[i])  # print to screen any path names that aren't covered
total_images_labels = total_images_labels.reshape((len(total_images),1))
print(total_images_labels.shape)


# Initialise new array to load images into
loaded_images = np.array([None]*len(total_images))
# loop through images, resizing and reshaping them and reassigning to new array
for i in range(len(total_images)):
    loaded_images[i] = cv2.resize(plt.imread(total_images[i]), (64, 64)).reshape((1, 64, 64, 3)) 

# concatenate all the images into the same array object
loaded_images = np.concatenate(loaded_images, axis=0)

# check the resulting shape 
print("Shape of Images array: ", loaded_images.shape)


# Perform the split of the data into train-test split 
train_images, test_images, train_labels, test_labels = train_test_split(loaded_images,
                    total_images_labels, test_size=0.25, random_state=101)

plt.imshow(train_images[4])
print(train_labels[4])

# convert our arrays to tensors
train_images = conv_tensor(train_images)
test_images = conv_tensor(test_images)
train_labels = conv_tensor(train_labels)
test_labels = conv_tensor(test_labels)



# Need to call the function in the model script
from Classify_Knee_Laterality_Model import classification



# evaluate and plot the stuff
# TODO need to pop this somewhere
images_shape = (64, 64, 3)

model = classification(images_shape)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics =['accuracy'])

# fit the keras model to the dataset
history = model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

