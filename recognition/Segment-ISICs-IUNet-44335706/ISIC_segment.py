'''
Build, train and test a TensorFlow based model to segment ISICs 2018 dermatology data
'''

# Import modules
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from IUNet_sequence import *
from IUNet_model import *

# Define global variables
X_DATA_LOCATION = "C:\\Users\\match\\Downloads\\ISIC2018_Task1-2_Training_Data\\ISIC2018_Task1-2_Training_Input_x2"
Y_DATA_LOCATION = "C:\\Users\\match\\Downloads\\ISIC2018_Task1-2_Training_Data\\ISIC2018_Task1_Training_GroundTruth_x2"
TRAIN_SIZE = 0.8
VALIDATE_SIZE = 0.1
TRAIN_BATCH_SIZE = 2
VALIDATE_BATCH_SIZE = 2
EPOCHS = 6


# Import data
train_seq, validate_seq, test_seq = split_data(X_DATA_LOCATION, Y_DATA_LOCATION, 'jpg', 'png', TRAIN_SIZE, VALIDATE_SIZE, TRAIN_BATCH_SIZE, VALIDATE_BATCH_SIZE)

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
    epochs=EPOCHS,
    validation_data=validate_seq,
)

# Plot the loss and accuracy curves of the training
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label ='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='upper left')

# Evaluate model
model.evaluate(test_seq)
