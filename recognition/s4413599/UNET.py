from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, Input, Dropout, MaxPooling2D, Conv2DTranspose, concatenate


IMG_SIZE = 128
IMG_CHANNEL = 1
BATCH_SIZE = 32

def get_Conv2D(filters, kernal_size=(3, 3), padding='same', activation='relu'):
    '''
    :param filters:
    :param kernal_size:
    :param padding:
    :param activation:
    :return:
    Help Function used to construct the Conv2d layer
    '''
    return Conv2D(filters, kernal_size, padding=padding, activation=activation)

def get_trs_Conv2D(filters, kernal_size=(2, 2), padding='same', strides=(2, 2)):
    '''
    :param filters:
    :param kernal_size:
    :param padding:
    :param strides:
    :return:
    Help Function used to construct the Transpose Conv2d layer
    '''
    return Conv2DTranspose(filters, kernal_size, strides=strides, padding='same')

def unet(IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL):
    '''
    :param IMG_SIZE: Size of the image
    :param IMG_CHANNEL: Number of image channel
    :return:
    '''
    inputs = Input((IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
    conv1 = get_Conv2D(16)(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = get_Conv2D(16)(conv1)
    pool1 = MaxPooling2D(2, 2)(conv1)

    conv2 = get_Conv2D(32)(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = get_Conv2D(32)(conv2)
    pool2 = MaxPooling2D(2, 2)(conv2)

    conv3 = get_Conv2D(64)(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = get_Conv2D(64)(conv3)
    pool3 = MaxPooling2D(2, 2)(conv3)

    conv4 = get_Conv2D(128)(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = get_Conv2D(128)(conv4)
    pool4 = MaxPooling2D(2, 2)(conv4)

    conv5 = get_Conv2D(256)(pool4)
    conv5 = Dropout(0.1)(conv5)
    conv5 = get_Conv2D(256)(conv5)

    uconv6 = get_trs_Conv2D(128)(conv5)
    uconv6 = concatenate([uconv6, conv4])
    conv6 = get_Conv2D(128)(uconv6)
    conv6 = Dropout(0.1)(conv6)
    conv6 = get_Conv2D(128)(conv6)

    uconv7 = get_trs_Conv2D(64)(conv6)
    uconv7 = concatenate([uconv7, conv3])
    conv7 = get_Conv2D(64)(uconv7)
    conv7 = Dropout(0.1)(conv7)
    conv7 = get_Conv2D(64)(conv7)

    uconv8 = get_trs_Conv2D(32)(conv7)
    uconv8 = concatenate([uconv8, conv2])
    conv8 = get_Conv2D(32)(uconv8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = get_Conv2D(32)(conv8)

    uconv9 = get_trs_Conv2D(64)(conv8)
    uconv9 = concatenate([uconv9, conv1], axis=3)
    conv9 = get_Conv2D(16)(uconv9)
    conv9 = Dropout(0.1)(conv9)

    conv9 = get_Conv2D(64)(conv9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    return Model(inputs=[inputs], outputs=[outputs])

# model = unet()
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# print(model.summary())