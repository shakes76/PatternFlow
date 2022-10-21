'''
    train.py
    Author: Jaydon Hansen
    Date created: 4/11/2020
    Date last modified: 7/11/2020
    Python Version: 3.8
'''

from math import floor
import os
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.engine import training
import numpy as np
import matplotlib.pyplot as plt

from model import UNet
from training_utils import *

img_size = 128
batch_size = 32
base_path = "/./data"
image_path = "./data/images"

training_ids = []
for file in os.walk(image_path):
    for filename in file:
        training_ids.append(filename)

# Get filenames and take file extensions off to leave IDs
training_ids = training_ids[2]
training_ids = [filename.split(".", 1)[0] for filename in training_ids]
print(training_ids)

# Split into training, test and validation data. 90% training, 10% test and 10% validation
validation_size = floor(len(training_ids) * 0.2)
test_size = floor(validation_size * 0.5)

# Split data into test, train validation
validation_test_ids = training_ids[:validation_size]
test_ids = validation_test_ids[:test_size]
validation_ids = validation_test_ids[test_size:]
training_ids = training_ids[validation_size:]

# Output to help with path debugging
print("Training images: ", len(training_ids))
print("Testing images: ", len(test_ids))
print("Validation images: ", len(validation_ids))

# Compile a model and show the shape
model = UNet(img_size)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef, "acc"])
model.summary()
plot_model(model, show_shapes=True)

# Create training and validation data generators
training_gen = DataGenerator(training_ids, base_path)
validation_gen = DataGenerator(validation_ids, base_path, 3, 128)

training_step_size = len(training_ids) // batch_size  # Step size of training data
validation_step_size = len(validation_ids) // batch_size  # Step size of validation data

history = model.fit_generator(
    training_gen,
    validation_data=validation_gen,
    steps_per_epoch=training_step_size,
    validation_steps=validation_step_size,
    epochs=20,
)
model.save("ISIC-UNet.h5") # Save the completed model

# Plot some example outputs for a sanity check
for i in range(0, 10):
    x, y = validation_gen.__getitem__(i)
    result = model.predict(x)

    result = result > 0.5  # make sure it's a well-predicted result
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(x[0])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.reshape(y[0] * 255, (img_size, img_size)), cmap="gray")
