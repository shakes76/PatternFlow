import sys
import os
import tensorflow as tf
from PIL import Image
import math
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, Dropout, LeakyReLU, BatchNormalization, UpSampling2D, concatenate, Add, Softmax
from tensorflow.keras import Model
from tensorflow.math import reduce_sum
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.image import resize_with_pad

num_epochs = argv[1]
batch_size = argv[2]
train_split = argv[3
val_split = argv[4]
test_split = argv[5]
dataset_size = argv[6]
image_height = argv[7]
image_width = argv[8]
x_data_location = argv[9]
y_data_location = argv[10]
if len(argv) != 8 or not math.isclose(float(train_split) + float(validation_split) + float(test_split), 1):
  print("Usage: Model.py [num_epochs] [batch_size] [train_split] [validation_split] [test_split] [dataset_size] [largest_image_height] [largest_image_width] [x_data_location] [y_data_location].\nPlease ensure train, validation and test split add up to 1 (e.g. 0.7, 0.15, 0.15)\nlargest_image_height and largest_image_width should be the largest height and width values of all the input images, respectively. This is to ensure all images are padded to the a valid size. Alternatively, you can just input very large values but this may cause problems")

#Load images into Tensorflow Datasets
x_dataset = tf.keras.utils.image_dataset_from_directory(x_data_location, labels=None)
y_dataset = tf.keras.utils.image_dataset_from_directory(y_data_location, labels=None)

#contains training data
x_train = x_dataset.take(math.floor(train_split * dataset_size))
y_train = y_dataset.take(math.floor(train_split * dataset_size))

#contains validation and test data. Is used simply to extract the validation and test parts.
val_and_test = x_dataset.skip(math.floor(train_split * dataset_size))
val_and_test = y_dataset.skip(math.floor(train_split * dataset_size))

#contains validation data
x_val = x_dataset.take(math.floor(val_split * dataset_size))
y_val = y_dataset.take(math.floor(val_split * dataset_size))

#contains test data
x_test = x_dataset.skip(math.floor(val_split * dataset_size))
y_test = y_dataset.skip(math.floor(val_split * dataset_size))

#normalise all images
def normalise(image):
  normalised_image = tf.math.divide(image, 255.0)
  return normalised_image

x_train = x_train.map(normalise)
y_train = y_train.map(normalise)

x_val = x_val.map(normalise)
y_val = y_val.map(normalise)

x_test = x_test.map(normalise)
y_test = y_test.map(normalise)

##pad images to be usable by Unet (both dimensions must be divisble by 81)

def pad_for_Unet(image):
  padded_image = resize_with_pad(image, image_height + height_padding, image_width + width_padding)
  return padded_image

height_padding = 0
width_padding = 0

if image_height%81 != 0:
  height_padding =  81 - (image_height%81)

if image_width%81 != 0:
  width_padding =  81 - (image_width%81)

x_train = x_train.map(pad_for_Unet)
y_train = y_train.map(pad_for_Unet)

x_val = x_val.map(pad_for_Unet)
y_val = y_val.map(pad_for_Unet)

x_test = x_test.map(pad_for_Unet)
y_test = y_test.map(pad_for_Unet)

#Defining Model Structure with Keras
base_channels = 16
dropout_rate = 0.3
leaky_relu_slope = 0.01
kernel_size = (3,3)
upsampling_kernel_size = (3,3)

#Defining Model Structure with Keras
base_channels = 4
dropout_rate = 0.3
leaky_relu_slope = 0.01
kernel_size = (3,3)
upsampling_kernel_size = (3,3)

input = Input(shape=(image_height + height_padding,image_width + width_padding,3))

#first descending section (16 channels by default)
initial_conv = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (input)
context1_conv1 = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (initial_conv)
context1_dropout = Dropout(dropout_rate) (context1_conv1)
context1_conv2 = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context1_dropout)
context1_batch_norm = BatchNormalization() (context1_conv2)

#second descending section (32 channels by default)
stride2_conv1 = Conv2D(base_channels * 2, (2,2), kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context1_batch_norm)
context2_conv1 = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv1)
context2_dropout = Dropout(dropout_rate) (context2_conv1)
context2_conv2 = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context2_dropout)
context2_batch_norm = BatchNormalization() (context2_conv2)

#third descending section (64 channels by default)
stride2_conv2 = Conv2D(base_channels * 4, (2,2), kernel_size, padding = "same",activation=LeakyReLU(leaky_relu_slope)) (context2_batch_norm)
context3_conv1 = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv2)
context3_dropout = Dropout(dropout_rate) (context3_conv1)
context3_conv2 = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context3_dropout)
context3_batch_norm = BatchNormalization() (context3_conv2)

#fourth descending section (128 channels by default)
stride2_conv3 = Conv2D(base_channels * 8, (2,2), kernel_size, padding = "same",activation=LeakyReLU(leaky_relu_slope)) (context3_batch_norm)
context4_conv1 = Conv2D(base_channels * 8, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv3)
context4_dropout = Dropout(dropout_rate) (context4_conv1)
context4_conv2 = Conv2D(base_channels * 8, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context4_dropout)
context4_batch_norm = BatchNormalization() (context4_conv2)

#bottom of "U" shape of the Unet - (256 channels by default)
stride2_conv4 = Conv2D(base_channels * 16, (2,2), kernel_size, padding = "same",activation=LeakyReLU(leaky_relu_slope)) (context4_batch_norm)
context5_conv1 = Conv2D(base_channels * 16, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv4)
context5_dropout = Dropout(dropout_rate) (context5_conv1)
context5_conv2 = Conv2D(base_channels * 16, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context5_dropout)
context5_batch_norm = BatchNormalization() (context5_conv2)
#first upsampling module
upsample1__upsample_layer = UpSampling2D(upsampling_kernel_size) (context5_batch_norm)
upsample1_conv = Conv2D(base_channels * 8, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (upsample1__upsample_layer)

concat1 = concatenate([context4_batch_norm, upsample1_conv])
#first localisation module
localisation1_conv1 = Conv2D(base_channels * 8, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (concat1)
localisation1_conv2 = Conv2D(base_channels * 8, (1,1), padding = "same", activation=LeakyReLU(leaky_relu_slope)) (localisation1_conv1)
#second upsampling module
upsample2_upsample_layer = UpSampling2D(upsampling_kernel_size) (localisation1_conv2)
upsample2_conv = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (upsample2_upsample_layer)

concat2 = concatenate([context3_batch_norm, upsample2_conv])
#second localisation module
localisation2_conv1 = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (concat2)
localisation2_conv2 = Conv2D(base_channels * 4, (1,1), padding = "same", activation=LeakyReLU(leaky_relu_slope)) (localisation2_conv1)
#third upsampling module
upsample3_upsample_layer = UpSampling2D(upsampling_kernel_size) (localisation2_conv2)
upsample3_conv = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (upsample3_upsample_layer)

concat3 = concatenate([context2_batch_norm, upsample3_conv])
#third localisation module
localisation3_conv1 = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (concat3)
localisation3_conv2 = Conv2D(base_channels * 2, (1,1), padding = "same", activation=LeakyReLU(leaky_relu_slope)) (localisation3_conv1)
#fourth upsampling module
upsample4_upsample_layer = UpSampling2D(upsampling_kernel_size) (localisation3_conv2)
upsample4_conv = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (upsample4_upsample_layer)

#final section
concat4 = concatenate([context1_batch_norm, upsample4_conv])
final_conv = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (concat4)

segmentation_layer1 = Conv2D(1, 1, activation=LeakyReLU(leaky_relu_slope)) (localisation2_conv2)

segmentation_layer1_upsampled = UpSampling2D(upsampling_kernel_size) (segmentation_layer1)
segmentation_layer2 = Conv2D(1, 1, activation=LeakyReLU(leaky_relu_slope)) (localisation3_conv2)

add1 = Add() ([segmentation_layer1_upsampled, segmentation_layer2])

add1_upsampled = UpSampling2D(upsampling_kernel_size) (add1)

segmentation_layer3 = Conv2D(1, 1, activation=LeakyReLU(leaky_relu_slope)) (final_conv)

add2 = Add() ([add1_upsampled, segmentation_layer3])

output = Conv2D(1, (1,1), padding = "same", activation="sigmoid") (add2)

unet = Model(input, output)
print(unet.summary())

def dice_coef_loss(true, predicted):
  #dice coefficient modified from: https://stackoverflow.com/questions/49785133/keras-dice-coefficient-loss-function-is-negative-and-increasing-with-epochs
    true_flat = tf.reshape(true, [-1])
    predicted_flat = tf.reshape(predicted, [-1])
    numerator = 2. * (reduce_sum(true_flat * predicted_flat) + 1.)
    denominator = (reduce_sum(true_flat) + reduce_sum(predicted_flat) + 1.)
    return 1. - numerator / denominator
    
#define dice coefficent loss function
callback = EarlyStopping(monitor='val_loss', patience = 2)
unet.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy'])

history = unet.fit(train, epochs = num_epochs, batch_size=batch_size, shuffle=True, validation_data=val, callbacks=callback)

predictions = unet.predict(test)
print(predictions[0][128])