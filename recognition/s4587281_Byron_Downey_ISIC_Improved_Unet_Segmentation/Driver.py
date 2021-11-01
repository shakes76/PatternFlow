from Model import create_model
import tensorflow as tf
from PIL import Image
import math
from tensorflow.image import resize
#from tensorflow.data.Dataset import zip
import matplotlib.pyplot as plt
from tensorflow.math import reduce_sum

num_epochs = 10
batch_size = 30
train_split = 0.80
val_split = 0.10
test_split = 0.10
x_data_location = "C:/Users/s4587281/Downloads/ISIC2018_Task1-2_Training_Input_x2"
y_data_location = "C:/Users/s4587281/Downloads/ISIC2018_Task1_Training_GroundTruth_x2"

input_shape = (324, 324, 3)
base_channels = 16
dropout_rate = 0.3
leaky_relu_slope = 0.03
kernel_size = (3,3)
upsampling_kernel_size = (3,3)

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

train = tf.data.Dataset.zip((x_train, y_train))
val = tf.data.Dataset.zip((x_val, y_val))
test = tf.data.Dataset.zip((x_test, y_test))

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
    
unet = create_model(num_epochs, batch_size, input_shape, base_channels, dropout_rate, leaky_relu_slope, kernel_size, upsampling_kernel_size)

unet.compile(optimizer=learning_rate_schedule, loss=dice_coef_loss)

history = unet.fit(train, epochs = num_epochs, batch_size=batch_size, shuffle=True, validation_data=val)

predictions = unet.predict(x_test)
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')

dice_coeff = [1. - datapoint for datapoint in history.history['loss']]
val_dice_coeff = [1. - datapoint for datapoint in history.history['val_loss']]
plt.figure()
plt.plot(dice_coeff, label='Training Dice Coefficient')
plt.plot(val_dice_coeff, label='Validation Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss = unet.evaluate(test, verbose=1)
print("Dice Similarity Coefficient: ", (1 - test_loss))

x_list = list(x_test)
y_list = list(y_test)
#displays 5 sets of images. Each set includes
for i in range(5):
    figure, (pos1, pos2, pos3) = plt.subplots(1, 3)
    pos1.imshow(x_list[i][0], cmap='gray')
    pos1.set_title('Original Image')
    pos1.set_xticks([])
    pos1.set_yticks([])

    pos2.imshow(y_list[i][0], cmap='gray')
    pos2.set_title('True Segmented Image')
    pos2.set_xticks([])
    pos2.set_yticks([])

    pos3.imshow(predictions[i], cmap='gray')
    pos3.set_title('Predicted Segmented Image')
    pos3.set_xticks([])
    pos3.set_yticks([])

    figure.tight_layout(w_pad=5)

plt.show()
