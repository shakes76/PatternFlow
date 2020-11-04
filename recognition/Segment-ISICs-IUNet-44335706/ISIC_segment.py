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
from IUNet_sequence import iunet_sequence

# Define global variables
X_DATA_LOCATION = 'C:/Users/match/Downloads/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'
Y_DATA_LOCATION = 'C:/Users/match/Downloads/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'
TRAIN_SIZE = 0.8
VALIDATE_SIZE = 0.1
TRAIN_BATCH_SIZE = 500
VALIDATE_BATCH_SIZE = 100


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
inputs = tf.keras.Input((256, 256, 1))

# Start block
layer = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
layer = tf.keras.layers.BatchNormalization()(layer)
previous_block = layer

# List for storing layers for concatenating
layers = []

# Create model
# Downsampling phase
for depth in (64, 128, 256):
	# First convolutional layer
	layer = tf.keras.layers.Conv2D(depth, 3, padding="same", activation="relu")(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)
	
	# Add dropout in between blocks to prevent overfitting
	layer = tf.keras.layers.Dropout(0.2)(layer)

	# Second convolutional layer
	layer = tf.keras.layers.Conv2D(depth, 3, padding="same", activation = "relu")(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)
	
	# Store layer in list
	layers.append(layer)
	
	# Pool down
	layer = tf.keras.layers.MaxPooling2D((2, 2))(layer)

# Upsampling phase
depths = (128, 64, 32)
for i in range(3):
	# First convolutional layer
	layer = tf.keras.layers.Conv2DTranspose(depths[i], 3, padding="same", activation="relu")(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)
	
	# Add dropout in between blocks to prevent overfitting
	layer = tf.keras.layers.Dropout(0.2)(layer)

	# Second convolutional layer
	layer = tf.keras.layers.Conv2DTranspose(depths[i], 3, padding="same", activation="relu")(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)
	
	# Normalise the batch and upsample
	
	layer = tf.keras.layers.UpSampling2D(2)(layer)
	layer = tf.keras.layers.Concatenate(axis=3)([layer, layers[-i-1]])

# Classify at the final output layer with softmax
outputs = tf.keras.layers.Conv2D(2, 3, activation="softmax", padding="same")(layer)

# Define the model
model = tf.keras.Model(inputs, outputs)
# print(model.summary())

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'])

# Train model
history = model.fit(
    train_seq,
    epochs=2,
    validation_data=validate_seq,
)

# Plot the loss and accuracy curves of the training
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label ='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='upper left')

# Evaluate model
model.evaluate(test_data)
