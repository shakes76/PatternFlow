import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU, Conv2D, Conv1D, MaxPooling2D, Dropout, UpSampling2D, UpSampling3D, \
    concatenate, Conv2DTranspose, Add
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


def unet_model():

    
    inputs = tf.keras.Input(shape=(256, 256, 3))
    conv1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(inputs)    
    conv1_1 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv1)
    pool1 = Dropout(0.3)(conv1_1)
    conv1_2 = Conv2D(16, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool1)
    add1 = Add()([conv1, conv1_2])
    
    conv2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(add1)
    conv2_1 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv2)
    pool2 = Dropout(0.3)(conv2_1)
    conv2_2 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool2)
    add2 = Add()([conv2, conv2_2])

    conv3 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(add2)
    conv3_1 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv3)
    pool3 = Dropout(0.3)(conv3_1)
    conv3_2 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool3)
    add3 = Add()([conv3, conv3_2])

    conv4 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(add3)
    conv4_1 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv4)
    pool4 = Dropout(0.3)(conv4_1)
    conv4_2 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool4)
    add4 = Add()([conv4, conv4_2])

    conv5= Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same', strides=(2, 2))(add4)
    conv5_1 = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv5)
    pool5 = Dropout(0.3)(conv5_1)
    conv5_2 = Conv2D(256, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(pool5)
    add5 = Add()([conv5, conv5_2])
    
    #Upsampling layers
    up1 = UpSampling2D()(add5)
    con1 = concatenate([up1, add4])
    up1_1 = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(con1)
    up1_2 = Conv2D(128, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(up1_1)

    up2 = UpSampling2D()(up1_2)
    con2 = concatenate([up2, add3])
    up2_1 = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(con2)
    up2_2 = Conv2D(64, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(up2_1)
    
    #Layers Segmentation
    seg1_1 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(up2_2)
    seg1_2 = UpSampling2D()(seg1_1)

    up3 = UpSampling2D()(up2_2)
    con3 = concatenate([up3, add2])
    up3_1 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(con3)
    up3_2 = Conv2D(32, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(up3_1)

    seg2 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(up3_2)

    up4 = UpSampling2D()(up3_2)
    con4 = concatenate([up4, add1])

    conv6 = Conv2D(32, (1, 1), activation=LeakyReLU(alpha=0.01), padding='same')(con4)

    seg3 = Conv2D(4, (3, 3), activation=LeakyReLU(alpha=0.01), padding='same')(conv6)

    #Combine Segmentation
    add_seg1 = Add()([seg1_2, seg2])
    add_seg2 = UpSampling2D()(add_seg1)
    add_seg3 = Add()([add_seg2, seg3])

    output_layer = Conv2D(1, 1, activation="sigmoid")(add_seg3)
    unetmodel = tf.keras.Model(inputs=inputs, outputs=output_layer)
    unetmodel.summary()
    return unetmodel
