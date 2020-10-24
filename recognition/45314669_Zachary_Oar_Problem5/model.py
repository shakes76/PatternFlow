from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
import math


def get_filenames_from_dir(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]


def encode_y(y):
    y = np.where(y < 0.5, 0, y)
    y = np.where(y > 0.5, 1, y)

    y = keras.utils.to_categorical(y, num_classes=2)
    return y


class SequenceGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batchsize):
        self.x, self.y, self.batchsize = x, y, batchsize

    def __len__(self):
        return math.ceil(len(self.x) / self.batchsize)

    def __getitem__(self, idx):
        x_names = self.x[idx * self.batchsize:(idx + 1) * self.batchsize]
        y_names = self.y[idx * self.batchsize:(idx + 1) * self.batchsize]
        
        # open x image names, resize, normalise and make a numpy array
        batch_x = np.array([np.asarray(Image.open("ISIC2018_Task1-2_Training_Input_x2/" 
                + file_name).resize((256, 192))) for file_name in x_names]) / 255.0

        # open y image names, resize, normalise, encode to one-hot and make a numpy array
        batch_y = np.array([np.asarray(Image.open("ISIC2018_Task1_Training_GroundTruth_x2/" 
                + file_name).resize((256, 192))) for file_name in y_names]) / 255.0
        batch_y = encode_y(batch_y)

        return batch_x, batch_y


x_names = get_filenames_from_dir("ISIC2018_Task1-2_Training_Input_x2")
y_names = get_filenames_from_dir("ISIC2018_Task1_Training_GroundTruth_x2")

# 15% of all the images are set aside as the test set
x_train_val, x_test, y_train_val, y_test = train_test_split(x_names, y_names, test_size=0.15, random_state=42)

# 17% of the non-test images are set aside as the validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.17, random_state=42)

train_gen = SequenceGenerator(x_train, y_train, 9)
sanity_check_x, sanity_check_y = train_gen.__getitem__(0)

# show some of the images as a sanity check
plt.figure(figsize=(10, 10))
for i in (0, 2, 4):
    plt.subplot(3, 2, i + 1)
    plt.imshow(sanity_check_x[i])
    plt.axis('off')

    plt.subplot(3, 2, i + 2)
    plt.imshow(tf.argmax(sanity_check_y[i], axis=2))
    plt.axis('off')
plt.show()