import sys
import os
import tensorflow as tf
from PIL import Image
import math
from tensorflow.keras.layers import Input, Conv2D, Dropout, LeakyReLU, BatchNormalization, UpSampling2D, concatenate, Add
from tensorflow.keras import Model
from tensorflow.math import reduce_sum
from tensorflow.image import resize
from tensorflow.data.Dataset import zip

num_epochs = argv[1]
batch_size = argv[2]
train_split = argv[3
val_split = argv[4]
test_split = argv[5]
x_data_location = argv[6]
y_data_location = argv[6]
if len(argv) != 8 or not math.isclose(float(train_split) + float(validation_split) + float(test_split), 1):
  print("Usage: Model.py [num_epochs] [batch_size] [train_split] [validation_split] [test_split] [x_data_location] [y_data_location].\nPlease ensure train, validation and test split add up to 1 (e.g. 0.7, 0.15, 0.15)")

#Load images into Tensorflow Datasets
x_dataset = tf.keras.utils.image_dataset_from_directory(x_data_location, batch_size=1,shuffle=False, labels=None)
y_dataset = tf.keras.utils.image_dataset_from_directory(y_data_location, batch_size=1, shuffle=False, labels=None)

dataset_size = len(list(x_dataset))

#contains training data
x_train = x_dataset.take(math.floor(train_split * dataset_size))
y_train = y_dataset.take(math.floor(train_split * dataset_size))

#contains validation and test data
x_val_and_test = x_dataset.skip(math.floor(train_split * dataset_size))
y_val_and_test = y_dataset.skip(math.floor(train_split * dataset_size))

#contains validation data
x_val = x_val_and_test.take(math.floor(val_split * dataset_size))
y_val = y_val_and_test.take(math.floor(val_split * dataset_size))

#contains test data
x_test = x_val_and_test.skip(math.floor(val_split * dataset_size))
y_test = y_val_and_test.skip(math.floor(val_split * dataset_size))

#resizes images to be usable by Unet (both dimensions must be divisble by 81)
def resize_for_Unet(image):
  padded_image= tf.image.resize(image, (324, 324))
  return padded_image

x_train = x_train.map(resize_for_Unet)
y_train = y_train.map(resize_for_Unet)

x_val = x_val.map(resize_for_Unet)
y_val = y_val.map(resize_for_Unet)

x_test = x_test.map(resize_for_Unet)
y_test = y_test.map(resize_for_Unet)

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

train = zip((x_train, y_train))
val = zip((x_val, y_val))
test = zip((x_test, y_test))

#Defining Model Structure with Keras layers
base_channels = 16
dropout_rate = 0.3
leaky_relu_slope = 0.01
kernel_size = (3,3)
upsampling_kernel_size = (3,3)

input = Input(shape=(324, 324,3))

#first downscaling section (16 channels by default)
initial_conv = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (input)
context1_conv1 = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (initial_conv)
context1_dropout = Dropout(dropout_rate) (context1_conv1)
context1_conv2 = Conv2D(base_channels, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context1_dropout)
context1_batch_norm = BatchNormalization() (context1_conv2)

#second downscaling section (32 channels by default)
stride2_conv1 = Conv2D(base_channels * 2, (2,2), kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context1_batch_norm)
context2_conv1 = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv1)
context2_dropout = Dropout(dropout_rate) (context2_conv1)
context2_conv2 = Conv2D(base_channels * 2, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context2_dropout)
context2_batch_norm = BatchNormalization() (context2_conv2)

#third downscaling section (64 channels by default)
stride2_conv2 = Conv2D(base_channels * 4, (2,2), kernel_size, padding = "same",activation=LeakyReLU(leaky_relu_slope)) (context2_batch_norm)
context3_conv1 = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (stride2_conv2)
context3_dropout = Dropout(dropout_rate) (context3_conv1)
context3_conv2 = Conv2D(base_channels * 4, kernel_size, padding = "same", activation=LeakyReLU(leaky_relu_slope)) (context3_dropout)
context3_batch_norm = BatchNormalization() (context3_conv2)

#fourth downscaling section (128 channels by default)
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

#upsamples previous segmentation layer so it has the same dimensions and can be added with the later one
segmentation_layer1_upsampled = UpSampling2D(upsampling_kernel_size) (segmentation_layer1)
segmentation_layer2 = Conv2D(1, 1, activation=LeakyReLU(leaky_relu_slope)) (localisation3_conv2)

#adds the segmentation layers and upsamples the result so it can be added with the last one
add1 = Add() ([segmentation_layer1_upsampled, segmentation_layer2])
add1_upsampled = UpSampling2D(upsampling_kernel_size) (add1)

segmentation_layer3 = Conv2D(1, 1, activation=LeakyReLU(leaky_relu_slope)) (final_conv)

#adds the final segmentation layer to the previous 2
add2 = Add() ([add1_upsampled, segmentation_layer3])

output = Conv2D(1, (1,1), padding = "same", activation=LeakyReLU(leaky_relu_slope)) (add2)
#sigmoid activation function, as there are only 2 classes represented by 0 and 1
output = tf.keras.layers.Dense(3, activation="sigmoid") (add2)

unet = Model(input, output)
print(unet.summary())

#dice coefficient loss function (simply 1 - dice coefficient) for use in training of model
def dice_coef_loss(true, predicted):
  #dice coefficient modified from: https://stackoverflow.com/questions/49785133/keras-dice-coefficient-loss-function-is-negative-and-increasing-with-epochs
    true_flat = tf.reshape(true, [-1])
    predicted_flat = tf.reshape(predicted, [-1])
    numerator = 2. * (reduce_sum(true_flat * predicted_flat) + 1.)
    denominator = (reduce_sum(true_flat) + reduce_sum(predicted_flat) + 1.)
    return 1. - numerator / denominator

smooth = 1.

learning_rate_schedule = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
unet.compile(optimizer=learning_rate_schedul, loss=dice_coef_loss)

history = unet.fit(train, epochs = num_epochs, batch_size=batch_size, shuffle=True, validation_data=val)