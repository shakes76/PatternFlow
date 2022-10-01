## Loading and preprocessing ISIC dataset

import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from modules import img_path, normalize

# Loading Data
test_img = keras.utils.image_dataset_from_directory(img_path.test, image_size=(256, 256), color_mode='grayscale')
val_img = keras.utils.image_dataset_from_directory(img_path.val, image_size=(256, 256), color_mode='grayscale')
train_img = keras.utils.image_dataset_from_directory(img_path.train, image_size=(256,256), color_mode='grayscale')

#Normalization
x_train, y_train = normalize(train_img)
x_val, y_val = normalize(val_img)
x_test, y_test = normalize(test_img)


#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(x_train[i])
#plt.show()