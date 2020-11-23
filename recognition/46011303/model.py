# -------------------------------------------------------------------------------------------------------------
# Author : Sivangi Mund
# Student id : 46011303
# Email : s.mund@uqconnect.edu.au
# Topic : Segmenting ISIC images using Improved Unet
#--------------------------------------------------------------------------------------------------------------






import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU, Conv2D, Conv1D, MaxPooling2D, Dropout, UpSampling2D, UpSampling3D, \
    concatenate, Conv2DTranspose, Add
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


def model():
    # taking input
    inputs = tensorflow.keras.Input(shape=(256, 256, 3))
    conv_layer = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(inputs)
    #context module
    conv_layer1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv_layer)
    pool1 = Dropout(0.3)(conv_layer1)
    conv_layer1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool1)
    #element wise summation
    conv_layer1 = Add()([conv_layer, conv_layer1])
    #downsampling layer between two context modules
    conv_layer_1s = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(conv_layer1)

    conv_layer2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv_layer_1s)
    pool2 = Dropout(0.3)(conv_layer2)
    conv_layer2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool2)
    conv_layer2 = Add()([conv_layer_1s, conv_layer2])

    conv_layer_2s = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(conv_layer2)

    conv_layer3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv_layer_2s)
    pool3 = Dropout(0.3)(conv_layer3)
    conv_layer3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool3)
    conv_layer3 = Add()([conv_layer_2s, conv_layer3])

    conv_layer_3s = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(conv_layer3)

    conv_layer4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv_layer_3s)
    pool4 = Dropout(0.3)(conv_layer4)
    conv_layer4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool4)
    conv_layer4 = Add()([conv_layer_3s, conv_layer4])

    conv_layer_4s = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(conv_layer4)

    conv_layer5 = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv_layer_4s)
    pool5 = Dropout(0.3)(conv_layer5)
    conv_layer5 = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool5)
    conv_layer5 = Add()([conv_layer_4s, conv_layer5])

    layer_6 = UpSampling2D()(conv_layer5)
     #concatenating with corresponding downsampling layer
    con1 = concatenate([layer_6, conv_layer4])

    up1 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(con1)
    up1 = Conv2D(128, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(up1)

    layer_7 = UpSampling2D()(up1)

    con2 = concatenate([layer_7, conv_layer3])

    up2 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(con2)
    up2 = Conv2D(64, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(up2)

    seg1 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(up2)
    seg1 = UpSampling2D()(seg1)

    layer_8 = UpSampling2D()(up2)

    con3 = concatenate([layer_8, conv_layer2])

    up3 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(con3)
    up3 = Conv2D(32, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(up3)

    seg2 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(up3)

    layer_9 = UpSampling2D()(up3)

    con4 = concatenate([layer_9, conv_layer1])

    layer_10 = Conv2D(32, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(con4)

    seg3 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(layer_10)

    #element wise summation of segmented layers
    added_seg12 = Add()([seg1, seg2])
    added_seg12 = UpSampling2D()(added_seg12)
    added_seg123 = Add()([added_seg12, seg3])

    conv_final = Conv2D(1, 1, activation="sigmoid")(added_seg123)
    model = tensorflow.keras.Model(inputs=inputs, outputs=conv_final)
    model.summary()
    return model