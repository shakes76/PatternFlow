## Loading and preprocessing ISIC dataset

import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from modules import img_path, normalize

# Loading Data
img_size = (448, 448)

test_img = keras.utils.image_dataset_from_directory(img_path.test, image_size=img_size, color_mode='grayscale')
val_img = keras.utils.image_dataset_from_directory(img_path.val, image_size=img_size, color_mode='grayscale')
train_img = keras.utils.image_dataset_from_directory(img_path.train, image_size=img_size, color_mode='grayscale')

#Normalization
x_train, y_train = normalize(train_img)
x_val, y_val = normalize(val_img)
x_test, y_test = normalize(test_img)

#Container
class data():
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

dataset = data(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test
)

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(dataset.x_train[i])
#plt.show()