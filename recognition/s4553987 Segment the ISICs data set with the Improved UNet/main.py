import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from random import shuffle, seed
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose
from tf.keras.callbacks import ModelCheckpoint

inputs = [cv2.imread(file) for file in glob.glob('D:\study\\7310\ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/*.jpg')]
outputs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob('D:\study\\7310\ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/*.png')]

for i in range(len(inputs)):
    inputs[i] = cv2.resize(inputs[i],(256,256))/255

for i in range(len(outputs)):
    outputs[i] = cv2.resize(outputs[i],(256,256))/255
    outputs[i][outputs[i] > 0.5] = 1
    outputs[i][outputs[i] <= 0.5] = 0

X = np.zeros([2594, 256, 256, 3])
y = np.zeros([2594, 256, 256])

for i in range(len(inputs)):
    X[i] = inputs[i]

for i in range(len(outputs)):
    y[i] = outputs[i]
        

y = y[:, :, :, np.newaxis]

X_train = X[0:1800,:,:,:]
X_val = X[1800:2197,:,:,:]
X_test = X[2197:2594,:,:,:]

y_train = y[0:1800,:,:,:]
y_val = y[1800:2197,:,:,:]
y_test = y[2197:2594,:,:,:]

model = improved_unet(256)

# Compile Model
def dsc(y_true, y_pred):
    intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred))
    dsc = 2*intersection / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return dsc

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dsc, 'accuracy'])


model_callback=tf.keras.callbacks.ModelCheckpoint(filepath='best_checkpoint',
                                              save_best_only= True,
                                              save_weights_only=True,
                                              monitor = 'val_accuracy',
                                              mode='max')
# train
history = model.fit(x = X_train, y=y_train, epochs=30, verbose=1,
                    validation_data=(X_val, y_val), batch_size = 8, callbacks=[model_callback])


# test
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

#plot the result
def plot_segment(model, X_test, y_test, dsc):
    """
    Plot input, groundtruth and prediction.
    """
    fig, ax = plt.subplots(3, 6, figsize = (16,8))
    for i in range(6):
        ax[0][i].imshow(X_test[i])
        ax[0][i].get_xaxis().set_visible(False)
        ax[0][i].get_yaxis().set_visible(False)
        ax[1][i].imshow(y_test[i])
        ax[1][i].get_xaxis().set_visible(False)
        ax[1][i].get_yaxis().set_visible(False)
        ax[2][i].imshow(tf.math.round(model.predict(X_test[i][np.newaxis,:,:,:]))[0])
        ax[2][i].get_xaxis().set_visible(False)
        ax[2][i].get_yaxis().set_visible(False)
        ax[2][i].set_title('dsc: '+str(round(dsc[i],2)))
        
image_dsc.sort()

a = 100 
high_dsc_images_X = [X_test[idx] for dsc,idx in image_dsc[-a:(-a-7):-1]]
high_dsc_images_y = [y_test[idx] for dsc,idx in image_dsc[-a:(-a-7):-1]]
high_dsc = [dsc for dsc,idx in image_dsc[-a:(-a-7):-1]]
plot_segment(model, high_dsc_images_X, high_dsc_images_y, high_dsc)

