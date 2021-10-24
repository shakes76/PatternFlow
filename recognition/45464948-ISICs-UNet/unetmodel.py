from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D , concatenate
from tensorflow.keras.models import Model

def model():

    #encoder part
    input_layer = Input(shape=(256,256,3))

    conv1 = Conv2D(64,(3,3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(64,(3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), padding='same')(conv1)
    conv2 = Conv2D(128,(3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128,(3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2), padding='same')(conv2)
    conv3 = Conv2D(256,(3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256,(3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2), padding='same')(conv3)
    conv4 = Conv2D(512,(3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512,(3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2), padding='same')(conv4)
    conv5 = Conv2D(1024,(3,3), activation='relu', padding='same')(pool4)
    encoded = Conv2D(1024,(3,3), activation='relu', padding='same')(conv5)

    #decoder part
    up_layer6 = UpSampling2D((2, 2))(encoded)
    conc6 = concatenate([conv4, up_layer6])
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conc6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    up_layer7 = UpSampling2D((2, 2))(conv6)
    conc7 = concatenate([conv3, up_layer7])
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conc7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    up_layer8 = UpSampling2D((2, 2))(conv7)
    conc8 = concatenate([conv2, up_layer8])
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    up_layer9 = UpSampling2D((2, 2))(conv8)
    conc9 = concatenate([conv1, up_layer8])
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    decoded = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    unetmodel = Model(input_layer,decoded)
    return unetmodel