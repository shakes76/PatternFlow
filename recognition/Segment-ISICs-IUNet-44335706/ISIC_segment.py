'''
Build, train and test a TensorFlow based model to segment ISICs 2018 dermatology data
'''

# Import modules
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
from math import ceil
from glob import glob
from IUNet_sequence import *
from IUNet_model import *

# Define global variables
X_DATA_LOCATION = 'H:\COMP3710\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1-2_Training_Input_x2'
Y_DATA_LOCATION = 'H:\COMP3710\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1_Training_GroundTruth_x2'
TRAIN_SIZE = 0.8
VALIDATE_SIZE = 0.1
TRAIN_BATCH_SIZE = 10
VALIDATE_BATCH_SIZE = 10
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# Import data
x_images = glob(X_DATA_LOCATION + '/*.jpg')
y_images = glob(Y_DATA_LOCATION + '/*.png')

x_images.sort()
y_images.sort()
x_images, y_images = shuffle(x_images, y_images)

train_index = ceil(len(x_images) * TRAIN_SIZE)
validate_index = ceil(len(x_images) * (TRAIN_SIZE + VALIDATE_SIZE))

train_seq = iunet_sequence(x_images[:train_index], y_images[:train_index], TRAIN_BATCH_SIZE)
validate_seq = iunet_sequence(x_images[train_index:validate_index], y_images[train_index:validate_index], VALIDATE_BATCH_SIZE)
test_seq = iunet_sequence(x_images[validate_index:], y_images[validate_index:], VALIDATE_BATCH_SIZE)

# Create model
model = iunet_model()
# print(model.summary())

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=dice_coef_loss,
    metrics=['accuracy'])

# Train model
history = model.fit(
    train_seq,
    epochs=6,
    validation_data=validate_seq,
)

# Plot the loss and accuracy curves of the training
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label ='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='upper left')
plt.show()

# Evaluate model
model.evaluate(test_seq)
