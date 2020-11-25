import os
import numpy as np
import glob
import skimage.io as io
import matplotlib.pyplot as plt
import random

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Setup the optimizer and loss function
smooth=1
def dice_coef(trainGen,testGen, smooth=1):
  intersection = K.sum(trainGen* testGen, axis=[1,2,3])
  union = K.sum(trainGen, axis=[1,2,3]) + K.sum(testGen, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def dice_coef_loss(trainGen,testGen):
  return -dice_coef(trainGen, testGen)

model.compile(optimizer=Adam(lr=1e-4), loss=-dice_coef_loss, metrics=[dice_coef])

# Set the proper checkpoint callback to save model.
model_checkpoint = ModelCheckpoint(
    'best_model.hdf5', monitor='loss', verbose=1, save_best_only=True)

# Model training.
history = model.fit_generator(
    trainGen,
    steps_per_epoch=train_steps,
    epochs=epochs,
    callbacks=[model_checkpoint],
    validation_data=valGen,
    validation_steps=validate_steps,
)

# Load the best model.
model = load_model('best_model.hdf5', compile=True)

# Model
example_image = "./data/keras_png_slices_test/case_441_slice_0.nii.png"
example_label = "./data/keras_png_slices_seg_test/seg_441_slice_0.nii.png"

# Read the images.
img = io.imread(example_image, as_gray=True)
img = img / 255.0
img = np.reshape(img, (1,) + img.shape)

# Get the predictions.
prob_predictions = model.predict(img)
final_predicitons = np.argmax(prob_predictions, axis=-1)

# Convert the one-hot predictions to image colors.
predictions = np.zeros(final_predicitons.shape, dtype=np.float32)
predictions[final_predicitons == 1] = 85.0
predictions[final_predicitons == 2] = 170.0
predictions[final_predicitons == 3] = 255.0

# Plot the image/prediction/groundtruth.
plt.subplot(131)
plt.imshow(img[0])
plt.title('Image')
plt.subplot(132)
plt.imshow(predictions[0])
plt.title('Prediction')
plt.subplot(133)
label = io.imread(example_label, as_gray=True)
plt.imshow(label)
plt.title('Groundtruth')
plt.show()

testGen = dataGenerator(
    1,
    './data/',
    'keras_png_slices_test',
    'keras_png_slices_seg_test',
    {},
    save_to_dir=None
)

predictions = []
labels = []
images = []

cnt = 0
for img, mask in testGen:
    cnt += 1
    if cnt > test_size:
        break
    
    # Get the images.
    images.append(img)
    
    # Get the predictions.
    prob_predictions = model.predict(img)
    final_predicitons = np.argmax(prob_predictions, axis=-1)
    final_labels = np.argmax(mask, axis=-1)
    predictions.append(final_predicitons)
    labels.append(final_labels)
    

predictions = np.stack(predictions, axis=0)
labels = np.stack(labels, axis=0)
predictions = np.squeeze(predictions, axis=1)
labels = np.squeeze(labels, axis=1)

print("Pixel accuracy per class: ")
PAs = []
for i in range(1, class_number):
    intersection = np.sum((predictions==labels) & (labels==i))
    union = np.sum(labels==i)
    PAs.append(intersection/union)
    print("Label {} accuracy: {}".format(i, intersection/union))
    
print()
print("mean pixel accuracy: ", np.mean(np.array(PAs)))

print("IOU per class: ")
IOUs = []
for i in range(1, class_number):
    intersection = np.sum((predictions==i) & (labels==i))
    union = np.sum((predictions==i) | (labels==i))
    IOUs.append(intersection/union)
    print("Label {} accuracy: {}".format(i, intersection/union))

print()
print("mean IOU: ", np.mean(np.array(IOUs)))

# Random select a image and its prediction and groundtruth.
i = random.choice(range(0, len(images)))

plt.subplot(131)
plt.imshow(images[i][0, :, :, 0])
plt.title('Image')
plt.subplot(132)
plt.imshow(predictions[0])
plt.title('Prediction')
plt.subplot(133)
plt.imshow(labels[0])
plt.title('Groundtruth')

plt.show()