"""

"""

import tensorflow as tf


from keras.models import Model
from keras.layers import Concatenate, Dropout, MaxPooling2D, Conv2D, LSTM, Input, concatenate, Cropping2D, Lambda, Conv2DTranspose


def data_loader():
    """ Loads in the ISIC training and ground-truth data
    """
    
    pass

def build_ISIC_cnn_model():
    """ Builds a unet
    """
    # ! Model inputs and normalisation
    # Input images are 511 x 384 x 3 (colour images)
    inputs = Input(shape=(511,384,3))
    crop = Cropping2D(cropping=((64,63),(0,0)))(inputs)
    s = Lambda(lambda x: x / 255)(crop)
    

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

    # ?
    # # Input shape
    # model.add(layers.Input(shape=(511,384)))

    # # Crop the image to turn the input into 384x384
    # model.add(layers.Cropping2D(cropping=((64,63),(0,0))))

    # # Normalize images values
    # model.add(layers.Lambda(lambda x: x / 255))
    
    # ! CONTRACTION PATH
    # * 24
    # kernel_intialiser sets the inital random weights that are to be trained.
    # model.add(layers.Conv2D(filters=24, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))

    # # Dropout for preventing overfitting
    # model.add(layers.Dropout(0.1))

    # model.add(layers.Conv2D(filters=24, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    
    # # Max pooling to halve the image resolution (W x H)
    # model.add(layers.MaxPool2D((2,2)))

    # * 48
    # model.add(layers.Conv2D(filters=48, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv2D(filters=48, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.MaxPool2D((2,2)))

    # # * 96
    # model.add(layers.Conv2D(filters=96, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv2D(filters=96, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.MaxPool2D((2,2)))

    # # * 192
    # model.add(layers.Conv2D(filters=192, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv2D(filters=192, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.MaxPool2D((2,2)))

    # # * 384
    # model.add(layers.Conv2D(filters=384, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv2D(filters=384, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))

    # # ! EXPANSIVE PATH
    # # * 192
    # model.add(layers.Conv2DTranspose(192, (2,2), strides=(2,2), padding='same'))
    # model.add(layers.Concatenate())
    # model.add(layers.Conv2D(filters=192, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Conv2D(filters=192, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))

    # # * 96
    # model.add(layers.Conv2DTranspose(96, (2,2), strides=(2,2), padding='same'))
    # model.add(layers.Concatenate())
    # model.add(layers.Conv2D(filters=96, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Conv2D(filters=96, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    
    # # * 48
    # model.add(layers.Conv2DTranspose(48, (2,2), strides=(2,2), padding='same'))
    # model.add(layers.Concatenate())
    # model.add(layers.Conv2D(filters=48, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Conv2D(filters=48, kernel_size=(3,3), 
    #         kernel_initializer='he_normal', activation='relu', padding='same'))

    # model.add(layers.Conv2D(1, (1,1), activation='sigmoid'))
if __name__ == '__main__':
    build_ISIC_cnn_model()