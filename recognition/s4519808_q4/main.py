"""
COMP3710 Report 

@author Huizhen 
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from random import shuffle, seed
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose

from model import *
from helper_functions import *
print("TensorFlow Version: ", tf.__version__) 

#### Parameters
H = W = 256
batch_size = 8


#### Data Process

# path
input_folder = './ISIC2018_Task1-2_Training_Input_x2'
output_folder = './ISIC2018_Task1_Training_GroundTruth_x2'
# image name
input_images = sorted([input_folder+'/'+file for file in os.listdir(input_folder) if file.endswith('jpg')])
output_images = sorted([output_folder+'/'+file for file in os.listdir(output_folder) if file.endswith('png')])
assert len(input_images)==len(output_images), 'input and output lists have different length'

# split into train, validatate and test
idx = list(range(len(input_images)))
seed(111)
shuffle(idx)
train_idx = idx[:1600]
validate_idx = idx[1600:2100]
test_idx = idx[2100:]
print(f'train, validate, test set have length: {len(train_idx)}, {len(validate_idx)}, {len(test_idx)}')

# train, test, validate images file name
train_input_images = [input_images[i] for i in train_idx]
train_output_images = [output_images[i] for i in train_idx]
validate_input_images = [input_images[i] for i in validate_idx]
validate_output_images = [output_images[i] for i in validate_idx]
test_input_images = [input_images[i] for i in test_idx]
test_output_images = [output_images[i] for i in test_idx]

# read images into numpy array
X_train = [resize_image(mpimg.imread(img)/255,H,W) for img in train_input_images]
X_train = np.array(X_train)
y_train = [resize_image(mpimg.imread(img)[:,:,np.newaxis],H,W) for img in train_output_images]
y_train = np.array(y_train)
X_val = [resize_image(mpimg.imread(img)/255,H,W) for img in validate_input_images]
X_val = np.array(X_val)
y_val = [resize_image(mpimg.imread(img)[:,:,np.newaxis],H,W) for img in validate_output_images]
y_val = np.array(y_val)
X_test = [resize_image(mpimg.imread(img)/255,H,W) for img in test_input_images]
X_test = np.array(X_test)
y_test = [resize_image(mpimg.imread(img)[:,:,np.newaxis],H,W) for img in test_output_images]
y_test = np.array(y_test)

#### Build Model
model = improved_unet(H, W)
#model.summary()

#### Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dsc, 'accuracy'])

#### Train
model_callback=tf.keras.callbacks.ModelCheckpoint(filepath='best_checkpoint',
                                              save_best_only= True,
                                              save_weights_only=True,
                                              monitor = 'val_accuracy',
                                              mode='max')
           
history = model.fit(x = X_train, y=y_train, epochs=30, verbose=1,
                    validation_data=(X_val, y_val), batch_size = batch_size, callbacks=[model_callback])


#### Test: Calculate Average Dice Similarity
image_dsc = []
s = 0
for i in range(len(test_input_images)):
        img = X_test[i][np.newaxis,:,:,:]
        pred = model.predict(img)
        predd = tf.math.round(pred)
        i_dsc = dsc(predd[0], y_test[i]).numpy()
        s += i_dsc
        image_dsc.append((i_dsc, i))
avg = s/len(test_input_images)
print('Average DSC: ',avg)

#### Plot
image_dsc.sort()

# good predictions
a = 100 
high_dsc_images_X = [X_test[idx] for dsc,idx in image_dsc[-a:(-a-7):-1]]
high_dsc_images_y = [y_test[idx] for dsc,idx in image_dsc[-a:(-a-7):-1]]
high_dsc = [dsc for dsc,idx in image_dsc[-a:(-a-7):-1]]
plot_segment(model, high_dsc_images_X, high_dsc_images_y, high_dsc)

# bad predictions
b = 10
low_dsc_images_X = [X_test[idx] for dsc,idx in image_dsc[b:(b+7)]]
low_dsc_images_y = [y_test[idx] for dsc,idx in image_dsc[b:(b+7)]]
low_dsc = [dsc for dsc,idx in image_dsc[b:(b+7)]]
plot_segment(model, low_dsc_images_X, low_dsc_images_y, low_dsc)