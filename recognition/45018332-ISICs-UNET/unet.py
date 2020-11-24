import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras import layers as krl

def model_unet(rows, cols, channels=1):
    #this is an exact implementation of the model described in (Ronneberger et al, 2015)
    #Input
    ins = kr.Input((rows,cols,channels))

    #downsample part of UNET
    #step 1
    conv1 = krl.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(ins)
    conv1 = krl.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conv1)
    mp1 = krl.MaxPool2D(pool_size=(2,2))(conv1)
    #step 2
    conv2 = krl.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(mp1)
    conv2 = krl.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conv2)
    mp2 = krl.MaxPool2D(pool_size=(2,2))(conv2)
    #step 3
    conv3 = krl.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(mp2)
    conv3 = krl.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(conv3)
    mp3 = krl.MaxPool2D(pool_size=(2,2))(conv3)
    #step 4
    conv4 = krl.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(mp3)
    conv4 = krl.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(conv4)
    mp4 = krl.MaxPool2D(pool_size=(2,2))(conv4)
    #step 5
    conv5 = krl.Conv2D(filters=1024, kernel_size=3, padding="same", activation="relu")(mp4)
    conv5 = krl.Conv2D(filters=1024, kernel_size=3, padding="same", activation="relu")(conv5)
    
    #upsampling part of UNET
    #step6
    ups6 = krl.Conv2DTranspose(filters=512, kernel_size=3, strides=(2,2), padding="same")(conv5)
    conc6 = krl.concatenate([conv4, ups6], axis=3)
    conv6 = krl.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(conc6)
    conv6 = krl.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu")(conv6)
    #step7
    ups7 = krl.Conv2DTranspose(filters=256, kernel_size=3, strides=(2,2), padding="same")(conv6)
    conc7 = krl.concatenate([conv3, ups7], axis=3)
    conv7 = krl.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(conc7)
    conv7 = krl.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(conv7)
    #step8
    ups8 = krl.Conv2DTranspose(filters=128, kernel_size=3, strides=(2,2), padding="same")(conv7)
    conc8 = krl.concatenate([conv2, ups8], axis=3)
    conv8 = krl.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conc8)
    conv8 = krl.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conv8)
    #step9
    ups9 = krl.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), padding="same")(conv8)
    conc9 = krl.concatenate([conv1, ups9], axis=3)
    conv9 = krl.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conc9)
    conv9 = krl.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conv9)

    outs = krl.Conv2D(filters=1, kernel_size=1, padding="same", activation="sigmoid")(conv9)
    
    model = kr.Model(inputs=ins, outputs=outs)

    return model