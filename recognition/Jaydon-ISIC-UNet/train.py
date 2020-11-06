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
base_path = "/content/data/images"
image_path = "/content/data/images/ISIC2018_Task1-2_Training_Input_x2"

training_ids = []
for file in os.walk(image_path):
    for filename in file:
        training_ids.append(filename)

# Get filenames and take file extensions off to leave IDs
training_ids = training_ids[2]
training_ids = [filename.split('.',1)[0] for filename in training_ids]
print(training_ids)

# Split into training, test and validation data
validation_size = floor(len(training_ids) * 0.2)
test_size = floor(validation_size * 0.5)

# Split data into test, train validation
validation_test_ids = training_ids[:validation_size]
test_ids = validation_test_ids[:test_size]
validation_ids = validation_test_ids[test_size:]
training_ids = training_ids[validation_size:]

#print("Total images: = ", len(test_ids + training_ids + validation_ids))
print("Training images: ", len(training_ids))
print("Testing images: ", len(test_ids))
print("Validation images: ", len(validation_ids))


# Instantiate a Generator
"""gen = DataGenerator(training_ids, base_path, batch_size=batch_size, image_size=img_size)

# Test the Generator
testX, testY = gen.__getitem__(0)
print("First image: ", testX.shape, testY.shape)

plot = plt.figure()
plot.subplots_adjust(hspace=0.5, wspace=0.5)
ax = plot.add_subplot(1, 2, 1)
ax.imshow(testX[0])
ax = plot.add_subplot(1, 2, 2)
ax.imshow(np.reshape(testY[0], (img_size, img_size)), cmap="gray")
"""

# Compile a model and show the shape
model = UNet(img_size, img_size)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef, "acc"])
model.summary()
plot_model(model, show_shapes=True)

# Create training and validation generators

training_gen = DataGenerator(training_ids, base_path)
validation_gen = DataGenerator(validation_ids, base_path, 3, 128)

training_step_size = len(training_ids) // batch_size # Step size of training data
validation_step_size = len(validation_ids) // batch_size # Step size of validation data

h = model.fit_generator(training_gen, validation_data = validation_gen, steps_per_epoch=training_step_size, validation_steps=validation_step_size, epochs = 20)
model.save("ISIC-UNet.h5")