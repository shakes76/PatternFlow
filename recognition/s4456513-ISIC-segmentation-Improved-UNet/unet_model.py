import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from tensorflow.keras import Model


def Improved_UNet(input_size):
    # input layer
    inputs = Input(input_size)

    # four VGG structures
    # first VGG
    conv1 = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv2 = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D((2,2))(conv2)

    # second VGG
    conv3 = Conv2D(128, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv4 = Conv2D(128, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool2 = MaxPooling2D((2,2))(conv4)
    drop1 = Dropout(0.5)(pool2)

    # third VGG
    conv5 = Conv2D(256, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(drop1)
    conv6 = Conv2D(256, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    pool3 = MaxPooling2D((2,2))(conv6)

    # forth VGG
    conv7 = Conv2D(512, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv8 = Conv2D(512, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    drop2 = Dropout(0.5)(conv8)
    pool4 = MaxPooling2D((2,2))(drop2)

    # bottom 
    conv9 = Conv2D(1024, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv10 = Conv2D(1024, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    # four CONCAT structures
    # first CONCAT
    up4 = UpSampling2D((2, 2), interpolation='nearest')(conv10)
    conv11 = Conv2D(512, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(up4)
    merge4 = Concatenate(axis=3)([conv8, up4])
    conv12 = Conv2D(512, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv13 = Conv2D(512, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)

    # second CONCAT
    up3 = UpSampling2D((2,2), interpolation='nearest')(conv13)
    conv14 = Conv2D(256, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(up3)
    merge3 = Concatenate(axis=3)([conv6, conv14])
    conv15 = Conv2D(256, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    conv16 = Conv2D(256, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv15)

    # third CONCAT
    up2 = UpSampling2D((2,2), iterpolation='nearest')(conv16)
    conv17 = Conv2D(128, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(up2)
    merge2 = Concatenate(axis=3)([conv4, conv17])
    conv18 = Conv2D(128, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    conv19 = Conv2D(128, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv18)

    # forth CONCAT
    up1 = UpSampling2D((2,2), iterpolation='nearest')(conv19)
    conv20 = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(up1)
    merge1 = Concatenate(axis=3)([conv2, conv20])
    conv21 = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    conv22 = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)

    # output 
    outputs = Conv2D(1,(1,1), activation='sigmod')(conv22)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model 