"""
    FILENAME: main.py

    AUTHOR: Harry Nowakowski

    DESCRIPTION: Implementation and testing of the Unet Improved problem.

    ACKNOWLEDGEMENTS: This method uses numpy in order to visualise data and load
        from the target data folder. It in no way uses numpy for anything else.

"""

import random
import sys
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.models import Model
from keras.layers import Concatenate, Dropout, MaxPooling2D, Conv2D, LSTM, \
    Input, concatenate, Cropping2D, Lambda, Conv2DTranspose, add
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K

def img_name_formatter(prefix: str, suffix: str, 
        imageNum: int, extension: str):
    """ Name formatting helper in order to iterate through the ISIC directory.

    ARGS:
    - prefix: The prefix we want on the front (e.g. 'ISIC_')
    - suffix: The suffix we want on the back (e.g. '_segmentation')
    - imageNum: The number of the image we want to pad with zeros
    - extension: e.g. 'jpg', 'png', etc.

    RETURNS:
    - The formatted image name.

    """
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

def image_plotter_helper(Train_X, Test_X):
    """ A small helper to plot a series of train and test images top to bottom. 
    
    ARGS:
    - Train_X: The actual images being fed into the model
    - Test_X: The expected/target output we want

    RETURNS:
    Nothing, it plots images :)
    """
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(Train_X[0])
    axarr[0,1].imshow(Train_X[1])
    axarr[1,0].imshow(Test_X[0])
    axarr[1,1].imshow(Test_X[1])
    plt.show()

def ISIC_data_loader(numTrain: int):
    """ Loads in the ISIC training and ground-truth data.

    Because the ISIC database contains only ~2900 files, yet it is indexed up to
    16k, I wanted to ensure people could use the original dataset directly with
    this program. Hence this program 'guesses' what the index of a particular
    file is iteratively.

    If the file exists in BOTH the train and test directory, then great, add it
    to the return arrays.

    If not, skip can find another one.

    ARGS:
    - numTrain: The number of training images we want.

    RETURN:
    - Returns two arrays of training and their corresponding test images.
    """
    minImg = 1
    maxImg = 16072
    targetSize = (384,384)
    # Some images (like 90.jpg) are missing, hence add a buffer
    # in case we need more
    buffer = abs(numTrain - maxImg)
 
    # Ensure that there are enough images
    if (numTrain + buffer) > maxImg:
        print("Not enough images in dataset: [{}] max".format(maxImg))
        sys.exit()

    training_pil_img_array = np.zeros((numTrain, 384, 384, 3), dtype=np.float32)

    # Since the data is either black or white, we interpret it as 
    test_pil_img_array = np.zeros((numTrain, 384, 384, 1), dtype=np.float32)

    choices = list(range(1,numTrain + buffer))
    choicesTrain = choices
    choicesTest = choices
    random.shuffle(choices)
    counter = 0
    while counter < numTrain:
        try:
            imgNumToTry = choices.pop()
            print("Loading training and testing images: {}%    ".format(round(counter/numTrain*100,2)), end="")
            imageTrain = load_img('data/ISIC2018_training_data/ISIC2018_Task1-2_Training_Input_x2/{}'\
                    .format(img_name_formatter('ISIC_', '', imgNumToTry, 'jpg')),
                    target_size=targetSize)
            imageTrain = img_to_array(imageTrain) / 255.0
            

            imageTest = load_img('data/ISIC2018_training_data/ISIC2018_Task1_Training_GroundTruth_x2/{}'\
                    .format(img_name_formatter('ISIC_', '_segmentation', imgNumToTry, 'png')),
                    target_size=targetSize,
                    color_mode='grayscale')
            imageTest = img_to_array(imageTest) / 255.0

            training_pil_img_array[counter,:,:,:] = imageTrain
            test_pil_img_array[counter,:,:,:] = imageTest

            counter += 1
            print("\r", end="")
        except FileNotFoundError as fnfe:
            print("\r", end="")
    print("\nDone")

    return (training_pil_img_array, test_pil_img_array)

def dice_coef(y_true, y_pred, smooth=1):
    """ Dice coefficient implementation. """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coeff_loss(y_true, y_pred):
    """ Dice coefficient loss implementation (nice for training models maybe). 
    """
    return 1-dice_coef(y_true, y_pred)

def build_ISIC_cnn_improved_model():
    # ! Model inputs and normalisation
    # Input images are 511 x 384 x 3 (colour images)
    inputs = Input((384,384,3))
    # crop = Cropping2D(cropping=((64,63),(0,0)))(inputs)
    # s = Lambda(lambda x: x / 255)(inputs)

    # ! Contraction path (first half of the 'U')
    # * 24
    con1Temp = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(inputs)

    con1 = Dropout(0.1)(con1Temp)
    con1 = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con1)
    con1 = add([con1, con1Temp])
    pool1 = MaxPooling2D((2,2))(con1)

    # * 48
    con2Temp = Conv2D(filters=48, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(pool1)
    con2 = Dropout(0.1)(con2Temp)
    con2 = Conv2D(filters=48, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con2)
    con2 = add([con2,con2Temp])
    pool2 = MaxPooling2D((2,2))(con2)

    # * 96
    con3Temp = Conv2D(filters=96, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(pool2)
    con3 = Dropout(0.1)(con3Temp)
    con3 = Conv2D(filters=96, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con3)
    con3 = add([con3, con3Temp])
    pool3 = MaxPooling2D((2,2))(con3)

    # * 192
    con4Temp = Conv2D(filters=192, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(pool3)
    con4 = Dropout(0.1)(con4Temp)
    con4 = Conv2D(filters=192, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con4)
    con4 = add([con4, con4Temp])
    pool4 = MaxPooling2D((2,2))(con4)

    # * 384
    con5Temp = Conv2D(filters=384, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(pool4)
    con5 = Dropout(0.1)(con5Temp)
    con5 = Conv2D(filters=384, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con5)
    con5 = add([con5, con5Temp])
    
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
    con8 = Dropout(0.1)(con8)
    con8 = Conv2D(filters=48, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con8)

    # ? Add two segmentation layers together
    seg1 = Conv2DTranspose(24, (4,4), strides=(4,4), padding='same')(ups7)
    seg2 = Conv2DTranspose(24, (2,2), strides=(2,2), padding='same')(ups8)
    segAdd = add([seg1, seg2])

    # * 24
    ups9 = Conv2DTranspose(24, (2,2), strides=(2,2), padding='same')(con8)
    ups9 = concatenate([ups9, con1])
    con9 = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(ups9)
    con9 = Dropout(0.1)(con9)
    con9 = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(con9)

    outSeg = add([segAdd, con9])

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(outSeg)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy', dice_coef])
    model.summary()
    return model

def build_ISIC_cnn_model():
    """ Builds a unet
    """
    # ! Model inputs and normalisation
    # Input images are 511 x 384 x 3 (colour images)
    inputs = Input(shape=(384,384,3))
    # crop = Cropping2D(cropping=((64,63),(0,0)))(inputs)
    # s = Lambda(lambda x: x / 255)(inputs)

    # ! Contraction path (first half of the 'U')
    # * 24
    con1 = Conv2D(filters=24, kernel_size=(3,3), 
            kernel_initializer='he_normal', activation='relu', padding='same')(inputs)
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
    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy', dice_coef])
    model.summary()
    return model

def train_unet():
    """ Trains the original unet model on the test set (note that the directory
        is configured for this test script and NOT google colab)
    """
    model = build_ISIC_cnn_model()

    # Model checkpoint
    checkpointer = ModelCheckpoint('ISIC_model_snapshot.h5', verbose=1, 
            save_freq=5)

    callbacks = [
    TensorBoard(log_dir='logs/regular_log_data')
    ]

    # Load data
    Train_X, Train_Y = ISIC_data_loader(1600)

    # plt.imshow(Train_X[0])
    # plt.show()

    # print("Train_X.shape = {}".format(Train_X.shape))
    # print("Test_X.shape = {}".format(Train_Y.shape))

    # #image_plotter_helper(Train_X, Train_Y)

    results = model.fit(Train_X, Train_Y, validation_split=0.1, batch_size=16, 
            epochs=100, callbacks=callbacks)

    model.save('models/Regular_model_2')

def test_unet():
    """ Runs the improved unet model with the a set of input data.
    """
    model = keras.models.load_model('models/Regular_model.h5', custom_objects={'dice_coef':dice_coef, 'dice_coeff_loss': dice_coeff_loss})

    targetSize = (384,384)

    imageTrain = load_img('data/ISIC2018_training_data/ISIC2018_Task1-2_Training_Input_x2/{}'\
            .format(img_name_formatter('ISIC_', '', 1, 'jpg')),
            target_size=targetSize)
    imageTrain = img_to_array(imageTrain) / 255.0

    imageTest = load_img('data/ISIC2018_training_data/ISIC2018_Task1_Training_GroundTruth_x2/{}'\
            .format(img_name_formatter('ISIC_', '_segmentation', 1, 'png')),
            target_size=targetSize,
            color_mode='grayscale')
    imageTest = img_to_array(imageTest) / 255.0

    plt.imshow(imageTrain)
    plt.show()

    plt.imshow(imageTest[0:,:,0])
    plt.show()

    origImg = imageTrain

    train_images1 = origImg[None,:,:,:]

    result = model.predict(train_images1)

    plt.imshow(result[0,:,:,0])
    plt.show()

def train_unet_improved():
    """ Trains the original unet architecture.
    """
    model = build_ISIC_cnn_improved_model()

    # Model checkpoint
    checkpointer = ModelCheckpoint('ISIC_model_snapshot_improved.h5', verbose=1, 
            save_freq=5)

    callbacks = [
        TensorBoard(log_dir='logs/improved_log_data')
    ]

    # Load data
    Train_X, Train_Y = ISIC_data_loader(1600)

    # plt.imshow(Train_X[0])
    # plt.show()

    # print("Train_X.shape = {}".format(Train_X.shape))
    # print("Test_X.shape = {}".format(Train_Y.shape))

    # #image_plotter_helper(Train_X, Train_Y)

    results = model.fit(Train_X, Train_Y, validation_split=0.1, batch_size=16, 
            epochs=100, callbacks=callbacks)

    model.save('/models/Improved_model_2')

def test_unet_improved():
    """ Runs the improved unet model with the a set of input data.
    """
    model = keras.models.load_model('models/Improved_model.h5', custom_objects={'dice_coef':dice_coef, 'dice_coeff_loss': dice_coeff_loss})

    targetSize = (384,384)

    imageTrain = load_img('data/ISIC2018_training_data/ISIC2018_Task1-2_Training_Input_x2/{}'\
            .format(img_name_formatter('ISIC_', '', 1, 'jpg')),
            target_size=targetSize)
    imageTrain = img_to_array(imageTrain) / 255.0

    imageTest = load_img('data/ISIC2018_training_data/ISIC2018_Task1_Training_GroundTruth_x2/{}'\
            .format(img_name_formatter('ISIC_', '_segmentation', 1, 'png')),
            target_size=targetSize,
            color_mode='grayscale')
    imageTest = img_to_array(imageTest) / 255.0

    plt.imshow(imageTrain)
    plt.show()

    plt.imshow(imageTest[0:,:,0])
    plt.show()

    origImg = imageTrain

    train_images1 = origImg[None,:,:,:]

    result = model.predict(train_images1)

    plt.imshow(result[0,:,:,0])
    plt.show()

if __name__ == '__main__':
    # Training functions
    train_unet()
    train_unet_improved()
    
    # Test functions
    test_unet()
    test_unet_improved()

    