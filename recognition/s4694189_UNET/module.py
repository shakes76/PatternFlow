
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, Conv2DTranspose, Dropout
from dataset import *

def unet_model():
    inputs = Input((256, 256, 3))
    x = inputs 
    # Contraction starts
    ## 1st downsampled network
    conv1 = Conv2D(16,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(x)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    
    ## 2nd downsampled network
    conv2 = Conv2D(32,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    
    ## 3rd downsampled network
    conv3 = Conv2D(64,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Conv2D(64,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    
    ## 4th downsampled network
    conv4 = Conv2D(128,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Conv2D(128,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    
    ## 5th downsampled network
    conv5 = Conv2D(256,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(pool4)
    conv5 = Dropout(0.1)(conv5)
    conv5 = Conv2D(256,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv5)
    pool5 = MaxPooling2D((2,2))(conv5)
    
    ##xpansion starts
    ## 1st upsampled network
    upconv6 = Conv2DTranspose(128,(2,2),strides = (2,2),padding="same")(conv5)
    upconv6 = concatenate([upconv6, conv4])
    conv6 = Conv2D(128,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(upconv6)
    conv6 = Dropout(0.1)(conv6)
    conv6 = Conv2D(128,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv6)
    
    ## 2nd upsampled network
    upconv7 = Conv2DTranspose(64,(2,2),strides = (2,2),padding="same")(conv6)
    upconv7 = concatenate([upconv7, conv3])
    conv7 = Conv2D(64,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(upconv7)
    conv7 = Dropout(0.1)(conv7)
    conv7 = Conv2D(64,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv7)
    
    ## 3rd upsampled network
    upconv8 = Conv2DTranspose(32,(2,2),strides = (2,2),padding="same")(conv7)
    upconv8 = concatenate([upconv8, conv2])
    conv8 = Conv2D(32,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(upconv8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv8)
    
    ## 4th upsampled network
    upconv9 = Conv2DTranspose(16,(2,2),strides = (2,2),padding="same")(conv8)
    upconv9 = concatenate([upconv9, conv1],axis=3)
    conv9 = Conv2D(16,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(upconv9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16,(3,3), padding="same", activation="relu", kernel_initializer="he_normal")(conv9)
    
    ##Output layer
    outputs = Conv2D(1,(1,1),activation="sigmoid")(conv9)

    cnn_model = Model(inputs = [inputs], outputs=[outputs])
    return cnn_model





