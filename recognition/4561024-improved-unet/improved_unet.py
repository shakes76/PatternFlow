'''
Improved UNet

@author Aghnia Prawira (45610240)
'''
import tensorflow as tf
from keras import backend as k
from keras.layers import Input, Conv2D, LeakyReLU, Dropout, Add, UpSampling2D, concatenate
from keras.models import Model

print('Tensorflow version:', tf.__version__)

def test():
    print("Testing improved unet.")
    
def unet():
    inputs = Input(shape=(256, 256, 1))
    
    c0 = Conv2D(16, (3, 3), padding='same')(inputs)
    c0 = LeakyReLU(alpha=0.01)(c0)
    
    '''Conv 1'''
    # Context module
    c1 = Conv2D(16, (3, 3), padding='same')(c0)
    c1 = LeakyReLU(alpha=0.01)(c1)
    c1 = Dropout(0.3)(c1)
    c1 = Conv2D(16, (3, 3), padding='same')(c1)
    c1 = LeakyReLU(alpha=0.01)(c1)
    # Element-wise sum
    c1 = Add()([c0, c1])
    
    # Downsampling
    c1_down = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(c1)
    c1_down = LeakyReLU(alpha=0.01)(c1_down)
    
    '''Conv 2'''
    # Context module
    c2 = Conv2D(32, (3, 3), padding='same')(c1_down)
    c2 = LeakyReLU(alpha=0.01)(c2)
    c2 = Dropout(0.3)(c2)
    c2 = Conv2D(32, (3, 3), padding='same')(c2)
    c2 = LeakyReLU(alpha=0.01)(c2)
    # Element-wise sum
    c2 = Add()([c1_down, c2])
    
    # Downsampling
    c2_down = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(c2)
    c2_down = LeakyReLU(alpha=0.01)(c2_down)
    
    '''Conv 3'''
    # Context module
    c3 = Conv2D(64, (3, 3), padding='same')(c2_down)
    c3 = LeakyReLU(alpha=0.01)(c3)
    c3 = Dropout(0.3)(c3)
    c3 = Conv2D(64, (3, 3), padding='same')(c3)
    c3 = LeakyReLU(alpha=0.01)(c3)
    # Element-wise sum
    c3 = Add()([c2_down, c3])
    
    # Downsampling
    c3_down = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(c3)
    c3_down = LeakyReLU(alpha=0.01)(c3_down)
    
    '''Conv 4'''
    # Context module
    c4 = Conv2D(128, (3, 3), padding='same')(c3_down)
    c4 = LeakyReLU(alpha=0.01)(c4)
    c4 = Dropout(0.3)(c4)
    c4 = Conv2D(128, (3, 3), padding='same')(c4)
    c4 = LeakyReLU(alpha=0.01)(c4)
    # Element-wise sum
    c4 = Add()([c3_down, c4])
    
    # Downsampling
    c4_down = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(c4)
    c4_down = LeakyReLU(alpha=0.01)(c4_down)
    
    '''Conv 5'''
    # Context module
    c5 = Conv2D(256, (3, 3), padding='same')(c4_down)
    c5 = LeakyReLU(alpha=0.01)(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), padding='same')(c5)
    c5 = LeakyReLU(alpha=0.01)(c5)
    # Element-wise sum
    c5 = Add()([c4_down, c5])
    
    # Upsampling module
    u4 = UpSampling2D(size=(2, 2))(c5)
    
    '''Up 4'''
    # Concatenation
    u4 = concatenate([u4, c4])
    # Localization module
    u4 = Conv2D(128, (3, 3), padding='same')(u4)
    u4 = LeakyReLU(alpha=0.01)(u4)
    u4 = Conv2D(128, (1, 1), padding='same')(u4)
    u4 = LeakyReLU(alpha=0.01)(u4)
    
    # Upsampling module
    u3 = UpSampling2D(size=(2, 2))(u4)
    
    '''Up 3'''
    # Concatenation
    u3 = concatenate([u3, c3])
    # Localization module
    u3 = Conv2D(64, (3, 3), padding='same')(u3)
    u3 = LeakyReLU(alpha=0.01)(u3)
    u3 = Conv2D(64, (1, 1), padding='same')(u3)
    u3 = LeakyReLU(alpha=0.01)(u3)
    # Segmentation module
    s3 = Conv2D(4, (3, 3), padding='same')(u3)
    s3 = LeakyReLU(alpha=0.01)(s3)
    s3 = UpSampling2D(size=(2, 2))(s3)
    
    # Upsampling module
    u2 = UpSampling2D(size=(2, 2))(u3)
    
    '''Up 2'''
    # Concatenation
    u2 = concatenate([u2, c2])
    # Localization module
    u2 = Conv2D(32, (3, 3), padding='same')(u2)
    u2 = LeakyReLU(alpha=0.01)(u2)
    u2 = Conv2D(32, (1, 1), padding='same')(u2)
    u2 = LeakyReLU(alpha=0.01)(u2)
    # Segmentation module
    s2 = Conv2D(4, (3, 3), padding='same')(u2)
    s2 = LeakyReLU(alpha=0.01)(s2)
    s3_2 = Add()([s3, s2])
    s3_2 = UpSampling2D(size=(2, 2))(s3_2)
    
    # Upsampling module
    u1 = UpSampling2D(size=(2, 2))(u2)
    
    '''Up 1'''
    # Concatenation
    u1 = concatenate([u1, c1])
    # Final conv layer
    u1 = Conv2D(32, (1, 1), padding='same')(u1)
    u1 = LeakyReLU(alpha=0.01)(u1)
    # Segmentation module
    s1 = Conv2D(4, (3, 3), padding='same')(u1)
    s1 = LeakyReLU(alpha=0.01)(s1)
    # Element-wise sum
    s3_2_1 = Add()([s3_2, s1])
    
    outputs = Conv2D(4, (3, 3), padding='same', activation='softmax')(s3_2_1)
    model = Model(inputs, outputs)
    model.summary()
    return model  

def dice_coefficient(y_true, y_pred):
    intersection = k.sum((y_true * y_pred), axis=[1,2,3])
    y_true_sum = k.sum(k.square(y_true), axis=[1,2,3])
    y_pred_sum = k.sum(k.square(y_pred), axis=[1,2,3])
    coefficient = 2 * intersection / (y_true_sum + y_pred_sum)
    return coefficient

def dice_coefficient_avg(y_true, y_pred):
    coefficient = k.mean(dice_coefficient(y_true, y_pred))
    return coefficient

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coefficient_avg(y_true, y_pred)
    return loss