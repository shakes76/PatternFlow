import tensorflow as tf
from tensorflow.keras import datasets, layers, models

""" This part goes in the driver script """
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Get filepaths
input_data_dir = "data\\reduced\\ISIC2018_Task1-2_Training_Input_x2"
gt_data_dir = "data\\reduced\\ISIC2018_Task1_Training_GroundTruth_x2"

input_data_paths = os.listdir(input_data_dir)
gt_data_paths = os.listdir(gt_data_dir)

if(len(input_data_paths) != len(gt_data_paths)):
    print("Mismatch in data / labels")

nTrain = len(input_data_paths)

testImg = Image.open(input_data_dir + "\\" + input_data_paths[0])
nx = 256
ny = 192
trainData = np.empty([nTrain, ny, nx, 3])

# iterate over data to construct full input data
for i, path in enumerate(input_data_paths):
    img = Image.open(input_data_dir + "\\" + path).resize((nx, ny))
    trainData[i] = np.asarray(img) / 255.0

nTypes = 2
trainLabels = np.empty([nTrain, ny, nx, nTypes])

# iterate over data to construct full input labels
for i, path in enumerate(gt_data_paths):
    img = Image.open(gt_data_dir + "\\" + path).resize((nx, ny))
    trainLabels[i] = np.eye(nTypes)[(np.asarray(img) * (1/255)).astype(int)]
#input_layer = layers.Input((256, 192, 3))

""" This part goes in the model script"""

### STANDARD U-NET
# down path

# down-level 1
input_layer = layers.Input((192,256,3))
conv_1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)
mPool_1 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_2)

# down-level 2
conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(mPool_1)
conv_4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_3)
mPool_2 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_4)

# down-level 3
conv_5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(mPool_2)
conv_6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_5)
mPool_3 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_6)

# down-level 4
conv_7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(mPool_3)
conv_8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_7)
mPool_4 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_8)

# bottom-level
conv_9 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(mPool_4)
conv_10 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv_9)

# up path
# up-level 4
uconv_1 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv_10)
cat_1 = layers.Concatenate()([uconv_1,conv_8])
conv_11 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(cat_1)
conv_12 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_11)

# up-level 3
uconv_2 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv_12)
cat_2 = layers.Concatenate()([uconv_2, conv_6])
conv_13 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(cat_2)
conv_14 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_13)

# up-level 2
uconv_3 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(conv_14)
cat_3 = layers.Concatenate()([uconv_3, conv_4])
conv_15 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cat_3)
conv_16 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_15)

# up-level 1
uconv_4 = layers.Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(conv_16)
cat_4 = layers.Concatenate()([uconv_4, conv_2])
conv_17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cat_4)
conv_18 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv_17)
conv_19 = layers.Conv2D(2, (1, 1), activation='softmax', padding='same')(conv_18)

unet = tf.keras.Model(inputs=input_layer, outputs=conv_19)
unet.compile(optimizer='adam', loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
print(unet.summary())

h = unet.fit(trainData, trainLabels, batch_size=16, epochs=10, validation_split=0.2)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])

samplePrediction_1 = unet.predict(trainData[0:1])

plt.figure()

plt.imshow(trainData[0])
plt.figure()
print(samplePrediction_1[0,:,:,:].shape)
plt.imshow(samplePrediction_1[0,:,:,0])
plt.figure()
plt.imshow(samplePrediction_1[0,:,:,1])
plt.figure()
plt.imshow(samplePrediction_1[0,:,:,:].argmax(axis=2))
plt.show()
