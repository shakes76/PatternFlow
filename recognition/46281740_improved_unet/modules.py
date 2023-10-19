from tensorflow.keras import backend as k
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Add, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def context_module(input, n: int):
    """
    Parameters:
        input -    input layer
        n     -    number of convolutional layers' kernel
    """
    # Context module
    c = Conv2D(n, (3, 3), padding='same')(input)
    c = LeakyReLU(alpha=0.01)(c)
    c = Dropout(0.3)(c)
    c = Conv2D(n, (3, 3), padding='same')(c)
    c = LeakyReLU(alpha=0.01)(c)
    return c

def down_sampling(input, n: int):
    """
    Parameters:
        input -    input layer
        n     -    number of convolutional layers' kernel
    """
    # Downsampling
    c = Conv2D(n, (3, 3), strides=(2, 2), padding='same')(input)
    c = LeakyReLU(alpha=0.01)(c)
    return c

def localization_model(input, n: int):
    """
    Parameters:
        input -    input layer
        n     -    number of convolutional layers' kernel
    """
    # Localization module
    c = Conv2D(n, (3, 3), padding='same')(input)
    c = LeakyReLU(alpha=0.01)(c)
    c = Conv2D(n, (1, 1), padding='same')(c)
    c = LeakyReLU(alpha=0.01)(c)
    return c

def segmentation_module(input, n: int, add: bool, add_layer=None):
    """
    Parameters:
        input     -    input layer
        n         -    number of convolutional layers' kernel
        add       -    if add is True then will apply an element-wise sum
        add_layer -    element-wise sum layer
    """
    # Segmentation module
    s = Conv2D(n, (3, 3), padding='same')(input)
    s = LeakyReLU(alpha=0.01)(s)
    if add is True:
        s = Add()([add_layer, s])
    s = UpSampling2D(size=(2, 2))(s)
    return s

def unet():
    inputs = Input(shape=(256, 256, 1))

    c0 = Conv2D(16, (3, 3), padding='same')(inputs)
    c0 = LeakyReLU(alpha=0.01)(c0)

    '''Conv 1'''
    c1 = context_module(input=c0, n=16)
    c1 = Add()([c0, c1])                      # Element-wise sum
    c1_down = down_sampling(input=c1, n=32)

    '''Conv 2'''
    c2 = context_module(input=c1_down, n=32)
    c2 = Add()([c1_down, c2])                 # Element-wise sum
    c2_down = down_sampling(input=c2, n=64)

    '''Conv 3'''
    c3 = context_module(input=c2_down, n=64)
    c3 = Add()([c2_down, c3])                 # Element-wise sum
    c3_down = down_sampling(input=c3, n=128)

    '''Conv 4'''
    c4 = context_module(input=c3_down, n=128)
    c4 = Add()([c3_down, c4])                 # Element-wise sum
    c4_down = down_sampling(input=c4, n=256)

    '''Conv 5'''
    c5 = context_module(input=c4_down, n=256)
    c5 = Add()([c4_down, c5])                 # Element-wise sum
    u4 = UpSampling2D(size=(2, 2))(c5)        # Upsampling module

    '''Up 4'''

    u4 = concatenate([u4, c4])                # Concatenation
    u4 = localization_model(input=u4, n=128)
    u3 = UpSampling2D(size=(2, 2))(u4)        # Upsampling module

    '''Up 3'''
    u3 = concatenate([u3, c3])                # Concatenation
    u3 = localization_model(input=u3, n=64)
    s3 = segmentation_module(input=u3, n=2, add=False)
    u2 = UpSampling2D(size=(2, 2))(u3)        # Upsampling module

    '''Up 2'''
    u2 = concatenate([u2, c2])                # Concatenation
    u2 = localization_model(input=u2, n=32)
    s3_2 = segmentation_module(input=u2, n=2, add=True, add_layer=s3)
    u1 = UpSampling2D(size=(2, 2))(u2)        # Upsampling module

    '''Up 1'''
    u1 = concatenate([u1, c1])                # Concatenation
    # Final conv layer
    u1 = Conv2D(32, (1, 1), padding='same')(u1)
    u1 = LeakyReLU(alpha=0.01)(u1)
    # Segmentation module
    s1 = Conv2D(2, (3, 3), padding='same')(u1)
    s1 = LeakyReLU(alpha=0.01)(s1)
    # Element-wise sum
    s3_2_1 = Add()([s3_2, s1])

    # Output
    outputs = Conv2D(2, (3, 3), padding='same', activation='softmax')(s3_2_1)
    model = Model(inputs, outputs)
    return model  

def dice_coefficient(y_true, y_pred):
    intersection = k.sum((y_true * y_pred), axis=[1,2,3])
    y_true_sum = k.sum(y_true, axis=[1,2,3])
    y_pred_sum = k.sum(y_pred, axis=[1,2,3])
    coefficient = (2.0 * intersection) / (y_true_sum + y_pred_sum)
    return coefficient

def dice_coefficient_avg(y_true, y_pred):
    coefficient = k.mean(dice_coefficient(y_true, y_pred))
    return coefficient

def dice_loss(y_true, y_pred):
    loss = 1.0 - dice_coefficient_avg(y_true, y_pred)
    return loss