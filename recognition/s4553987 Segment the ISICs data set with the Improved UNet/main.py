import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.image import resize
from random import shuffle, seed
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose
from tf.keras.callbacks import ModelCheckpoint
from model import*

# load data
inputs = sorted([input_folder+'/'+file for file in os.listdir(./ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2) if file.endswith('jpg')])
outputs = sorted([output_folder+'/'+file for file in os.listdir(./ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2) if file.endswith('png')])

# resize data and split the data into train dataset, test dataset and validation dataset
def resize_data(inputs):  
    outputs = [resize(mpimg.imread(data)/255,[256,256]).numpy() for data in inputs]
    outputs = np.array(outputs)
    return outputs

X_train_data = inputs[0:1800,:,:,:]
X_val_data = inputs[1800:2197,:,:,:]
X_test_data = inputs[2197:2594,:,:,:]

y_train_data = outputs[0:1800,:,:,:]
y_val_data = outputs[1800:2197,:,:,:]
y_test_data = outputs[2197:2594,:,:,:]

X_train =  resize_data(X_train_data)
y_train = resize_data(y_train_data)
X_val = resize_data(X_val_data)
y_val = resize_data(y_val_data)
X_test = resize_data(X_test_data)
y_test = resize_data(y_test_data)

    
# Compile Model
model = improved_u_net(256)

def dsc(y_true, y_pred):
    intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred))
    dsc = 2*intersection / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return dsc

if __name__ == "__main__":
    model = improved_u_net(256)
    metric = [dsc, 'accuracy']
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics= metric)
    model_checkpoint = ModelCheckpoint(filepath='best_checkpoint',save_best_only= True,save_weights_only=True,monitor = 'val_accuracy',mode='max')
    # train
    history = model.fit(x = X_train, y=y_train, epochs=30, verbose=1,validation_data=(X_val, y_val), batch_size = 8, callbacks=[model_checkpoint])
    
# test by average DSC
data_dsc = []
dsc_sum = 0
l = len(y_test_data)
for i in range(l):
        data = X_test[i][np.newaxis,:,:,:]
        prediction = model.predict(data)
        prediction_r = tf.math.round(prediction)
        dsc_i = dsc(prediction_r[0], y_test_data[i]).numpy()
        dsc_sum += dsc_i
        data_dsc.append((dsc_i, i))
dsc_avg = dsc_sum/l
print('Average DSC: ',dsc_avg)

#plot the pic for train data and test data
def plot_segment(model, X_test, y_test, dsc):
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
        
 
images_X = [X_test[idx] for dsc,idx in data_dsc[-200:(-207):-1]]
images_y = [y_test[idx] for dsc,idx in data_dsc[-200:(-207):-1]]
dsc = [dsc for dsc,idx in data_dsc[-200:(-207):-1]]
plot_segment(model, images_X, images_y, dsc)

