"""
A Dcgan project about generating brain slice pictures
And finally the structure similarity is over 0.6

@author: Amanda
@time : 30/10/2020
"""

from model import * 

# libraries used in the project 

import glob,cv2,time
from PIL import Image
import matplotlib.pyplot as plt 
from IPython import display
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import os

# load data 
images_train = [cv2.imread(file) for file in glob.glob("keras_png_slices_data/keras_png_slices_train/*.png")]
images_test = [cv2.imread(file) for file in glob.glob("keras_png_slices_data/keras_png_slices_test/*.png")]

# know the shape of the data
print("X_train number of samples = ",len(images_train))
print("X_test number of samples ", len(images_test))
print("original image_shape = ", images_test[0].shape)

# show how the images look like 
fig = plt.figure(figsize=(8,8))
for i in range(4):
    ax = fig.add_subplot(2,2,i+1)
    ax.imshow(images_train[i+1])
    ax.axis('off')
plt.show()


# Change the RGB into GRAY
X_test = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images_test]
X_train = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images_train]

#Convert to tensor and normalise into -1 and 1
tf_test = tf.convert_to_tensor(X_test,dtype=tf.float32)
tf_train = tf.convert_to_tensor(X_train,dtype=tf.float32)

print("the minimum of train_images ",tf.reduce_min(tf_train).numpy())
print("the maximum of train_images ",tf.reduce_max(tf_train).numpy())
print("the minimum of test_images ",tf.reduce_min(tf_test).numpy())
print("the maximum of test_images ",tf.reduce_max(tf_test).numpy())

# Normalise the data into -1 and 1
# the formula is data-127.5/127.5
half_max1 = tf.math.scalar_mul(127.5, tf.ones_like(tf_test,dtype=tf.float32))
temp_test = tf.math.subtract(tf_test, half_max1)
tf_test = tf.math.scalar_mul(1/127.5, temp_test)
tf_test = tf.reshape(tf_test, [544, 64, 64, 1])

half_max2 = tf.math.scalar_mul(127.5, tf.ones_like(tf_train,dtype=tf.float32))
temp_train = tf.math.subtract(tf_train, half_max2)
tf_train = tf.math.scalar_mul(1/127.5, temp_train)
tf_train = tf.reshape(tf_train, [9664, 64, 64, 1])

#check how images how like now
fig = plt.figure(figsize=(8,8))
for i in range(4):
    ax = fig.add_subplot(2,2,i+1)
    ax.imshow(tf_train[i+1])
    ax.axis('off')
plt.show()

# Shuffle and batch data
BUFFER_SIZE = tf_train.shape[0]
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(tf_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#train the model and return the max ssim for each epoch
ssim = train(train_dataset, EPOCHS)

plt.plot(range(len(ssim)), ssim)
plt.title('Struture Similarity vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Ssim ')
plt.savefig("ssim.png",dpi = 700)