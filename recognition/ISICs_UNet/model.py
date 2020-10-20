from tensorflow.keras import models, layers, Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, UpSampling2D, MaxPooling2D, concatenate

## Model for the UNet from https://arxiv.org/abs/1505.04597
def unet(input_size=(256,256,1)):
    INIT_FILTER = 32
    hn = 'he_normal'
    dropout = 0.2

    inputs = Input(input_size)
    
    conv1 = Conv2D(INIT_FILTER * 1, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(inputs)
    conv1 = Conv2D(INIT_FILTER * 1, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(INIT_FILTER * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(pool1)
    conv2 = Conv2D(INIT_FILTER * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(INIT_FILTER * 3, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(pool2)
    conv3 = Conv2D(INIT_FILTER * 3, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(INIT_FILTER * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(pool3)
    conv4 = Conv2D(INIT_FILTER * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(INIT_FILTER * 5, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(pool4)
    conv5 = Conv2D(INIT_FILTER * 5, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(INIT_FILTER * 4, (2,2), activation = 'relu', padding = 'same', kernel_initializer=hn)(UpSampling2D(size = (2,2))(drop5))
    concat6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(INIT_FILTER * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(concat6)
    conv6 = Conv2D(INIT_FILTER * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv6)

    up7 = Conv2D(INIT_FILTER * 3, (2,2), activation = 'relu', padding = 'same', kernel_initializer=hn)(UpSampling2D(size = (2,2))(conv6))
    concat7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(INIT_FILTER * 3, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(concat7)
    conv7 = Conv2D(INIT_FILTER * 3, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv7)

    up8 = Conv2D(INIT_FILTER * 2, (2,2), activation = 'relu', padding = 'same', kernel_initializer=hn)(UpSampling2D(size = (2,2))(conv7))
    concat8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(INIT_FILTER * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(concat8)
    conv8 = Conv2D(INIT_FILTER * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv8)

    up9 = Conv2D(INIT_FILTER * 1, (2,2), activation = 'relu', padding = 'same', kernel_initializer=hn)(UpSampling2D(size = (2,2))(conv8))
    concat9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(INIT_FILTER * 1, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(concat9)
    conv9 = Conv2D(INIT_FILTER * 1, (3,3), activation = 'relu', padding = 'same', kernel_initializer=hn)(conv9)

    outputs = Conv2D(4, (1,1), activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    # model.summary()

    return model


## Model for the improved UNet from https://arxiv.org/abs/1802.10508v1
def improved_unet():
    return None
    