"""

"""

import tensorflow as tf
import random
import sys
# Pyplot for having a look at our image data
import matplotlib.pyplot as plt

# ! NOTE - Using numpy for importing images (NOTHING ELES)
import numpy as np

from keras.models import Model
from keras.layers import Concatenate, Dropout, MaxPooling2D, Conv2D, LSTM, Input, concatenate, Cropping2D, Lambda, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import load_img, img_to_array

def img_name_formatter(prefix: str, suffix: str, 
        imageNum: int, extension: str):
    if imageNum >= 1 and imageNum <= 9:
        return "{}000000{}{}.{}".format(prefix, imageNum, suffix, extension)
    elif imageNum >= 10 and imageNum <= 99:
        return "{}00000{}{}.{}".format(prefix, imageNum, suffix, extension)
    elif imageNum >= 100 and imageNum <= 999:
        return "{}0000{}{}.{}".format(prefix, imageNum, suffix, extension)
    elif imageNum >= 1000 and imageNum <= 9999:
        return "{}000{}{}.{}".format(prefix, imageNum, suffix, extension)
    elif imageNum >= 10000 and imageNum <= 99999:
        return "{}00{}{}.{}".format(prefix, imageNum, suffix, extension)
    elif imageNum >= 100000 and imageNum <= 999999:
        return "{}0{}{}.{}".format(prefix, imageNum, suffix, extension)

def image_plotter_helper(Train_X):
    fig = plt.figure(figsize=(30,10))
    nplot = 7
    for count in range(1,nplot):
        ax = fig.add_subplot(1,nplot,count)
        ax.imshow(Train_X[count])
    plt.show()

def ISIC_data_loader(numTrain: int, numTest: int):
    """ Loads in the ISIC training and ground-truth data
    """
    minImg = 1
    maxImg = 16072
    targetSize = (384,384)
    # Some images (like 90.jpg) are missing, hence add a buffer
    # in case we need more
    buffer = 10000
 
    # Ensure that there are enough images
    if (numTrain + numTest) > maxImg:
        print("Not enough images in dataset: [{}] max".format(maxImg))
        sys.exit()

    training_pil_img_array = np.zeros((numTrain, 384, 384, 3), dtype=np.uint8)

    # Since the data is either black or white, we interpret it as 
    test_pil_img_array = np.zeros((numTest, 384, 384, 1), dtype=np.bool)

    choices = list(range(1,numTrain + numTest + buffer))
    random.shuffle(choices)
    counter = 0
    while counter < numTrain:
        try:
            print("Loading training images: {}%    ".format(round(counter/numTrain*100,2)), end="")
            image = load_img('data/ISIC2018_training_data/ISIC2018_Task1-2_Training_Input_x2/{}'\
                    .format(img_name_formatter('ISIC_', '', choices.pop(), 'jpg')),
                    target_size=targetSize)
            image = img_to_array(image)
            training_pil_img_array[counter,:,:,:] = image
            counter += 1
            print("\r", end="")
        except FileNotFoundError as fnfe:
            print("\r", end="")
    print("\nDone")

    counter = 0
    while counter < numTest:
        try:
            print("Loading test images: {}%    ".format(round(counter/numTest*100,2)), end="")
            image = load_img('data/ISIC2018_training_data/ISIC2018_Task1_Training_GroundTruth_x2/{}'\
                    .format(img_name_formatter('ISIC_', '_segmentation', choices.pop(), 'png')),
                    target_size=targetSize,
                    color_mode='grayscale')
            image = img_to_array(image)
            test_pil_img_array[counter,:,:,:] = image
            counter += 1
            print("\r", end="")
        except FileNotFoundError as fnfe:
            print("\r", end="")
    print("\nDone")

    return (training_pil_img_array, test_pil_img_array)

def build_ISIC_cnn_model():
    """ Builds a unet
    """
    # ! Model inputs and normalisation
    # Input images are 511 x 384 x 3 (colour images)
    inputs = Input(shape=(511,384,3))
    crop = Cropping2D(cropping=((64,63),(0,0)))(inputs)
    s = Lambda(lambda x: x / 255)(inputs)
    

    # ! Contraction path (first half of the 'U')
    # * 24
    con1 = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(s)
    con1 = Dropout(0.1)(con1)
    con1 = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con1)
    pool1 = MaxPooling2D((2,2))(con1)

    # * 48
    con2 = Conv2D(filters=48, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(pool1)
    con2 = Dropout(0.1)(con2)
    con2 = Conv2D(filters=48, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con2)
    pool2 = MaxPooling2D((2,2))(con2)

    # * 96
    con3 = Conv2D(filters=96, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(pool2)
    con3 = Dropout(0.1)(con3)
    con3 = Conv2D(filters=96, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con3)
    pool3 = MaxPooling2D((2,2))(con3)

    # * 192
    con4 = Conv2D(filters=192, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(pool3)
    con4 = Dropout(0.1)(con4)
    con4 = Conv2D(filters=192, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con4)
    pool4 = MaxPooling2D((2,2))(con4)

    # * 384
    con5 = Conv2D(filters=384, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(pool4)
    con5 = Dropout(0.1)(con5)
    con5 = Conv2D(filters=384, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con5)
    
    # ! Expansive path (second half of the 'U')

    # * 192
    ups6 = Conv2DTranspose(192, (2,2), strides=(2,2), padding='same')(con5)
    ups6 = concatenate([ups6, con4])
    con6 = Conv2D(filters=192, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(ups6)
    con6 = Dropout(0.2)(con6)
    con6 = Conv2D(filters=192, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con6)
    
    # * 96
    ups7 = Conv2DTranspose(96, (2,2), strides=(2,2), padding='same')(con6)
    ups7 = concatenate([ups7, con3])
    con7 = Conv2D(filters=96, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(ups7)
    con7 = Dropout(0.2)(con7)
    con7 = Conv2D(filters=96, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con7)

    # * 48
    ups8 = Conv2DTranspose(48, (2,2), strides=(2,2), padding='same')(con7)
    ups8 = concatenate([ups8, con2])
    con8 = Conv2D(filters=48, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(ups8)
    con8 = Dropout(0.2)(con8)
    con8 = Conv2D(filters=48, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con8)

    # * 24
    ups9 = Conv2DTranspose(24, (2,2), strides=(2,2), padding='same')(con8)
    ups9 = concatenate([ups9, con1])
    con9 = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(ups9)
    con9 = Dropout(0.2)(con9)
    con9 = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(con9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
if __name__ == '__main__':
    # model = build_ISIC_cnn_model()

    # # Model checkpoint
    # checkpointer = ModelCheckpoint('ISIC_model_snapshot.h5', verbose=1, 
    #         save_freq=5)

    # callbacks = [
    #     EarlyStopping(patience=2, monitor='val_loss'),
    #     TensorBoard(log_dir='logs/log_file')
    # ]

    # Load data
    Train_X, Test_X = ISIC_data_loader(500, 100)

    print("Train_X.shape = {}".format(Train_X.shape))
    print("Test_X.shape = {}".format(Test_X.shape))

    image_plotter_helper(Train_X)

    # results = model.fit(X,Y, validation_split=0.1, batch_size=16, epochs=25, 
    #         callbacks=callbacks)

    