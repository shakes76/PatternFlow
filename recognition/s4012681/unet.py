import numpy as np
import tensorflow as tf
import nibabel
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, BatchNormalization, concatenate, Dropout, \
    MaxPooling3D, Dense
from tensorflow.keras import Model

# Labels:
# Background = 0
# Body = 1
# Bones = 2
# Bladder = 3
# Rectum = 4
# Prostate = 5

IMG_WIDTH = 128
IMG_HEIGHT = 256
IMG_DEPTH = 256
IMG_CHANNELS = 1


def get_nifti_data(file_name):
    # tf.string(file_name)
    # bits = tf.io.read_file(file_name)
    #print(file_name)
    img = nibabel.load(file_name).get_fdata()
    return img


def one_hot(file_name):
    mask = get_nifti_data(file_name)
    bg = mask == 0
    bg = tf.where(bg == True, 1, 0)
    body = mask == 1
    body = tf.where(body == True, 1, 0)
    bones = mask == 2
    bones = tf.where(bones == True, 1, 0)
    bladder = mask == 3
    bladder = tf.where(bladder == True, 1, 0)
    rectum = mask == 4
    rectum = tf.where(rectum == True, 1, 0)
    prostate = mask == 5
    prostate = tf.where(prostate == True, 1, 0)
    return tf.concat((bg, body, bones, bladder, rectum, prostate), axis=-1)


def normalise(image):
    # subtract mean
    # mean = np.average(image)
    # image = image - mean

    # divide by sd
    # sd = np.std(image)
    # image = image / sd

    # divide by max value
    maximum = np.max(image)
    image = image / maximum
    return image


def reshape(batch_size, dimension, image):
    return np.reshape(image, (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, dimension))


def reshape_mask(batch_size, dimension, image):
    return np.reshape(image, (batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, dimension))


def map_fn(image, mask):
    image = get_nifti_data(image)
    mask = one_hot(mask)
    return image, mask

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def weighted_cross(beta):
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)

        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        return tf.reduce_mean(o)
    return loss

def unet(filters):
    inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, IMG_CHANNELS))

    # Contraction
    c1 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling3D((2, 2, 2)) (c1)

    c2 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling3D((2, 2, 2)) (c2)

    c3 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling3D((2, 2, 2)) (c3)

    c4 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2)) (c4)

    c5 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    # Expansion
    u6 = Conv3DTranspose(filters * 16, (2, 2, 2), strides=(2, 2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv3DTranspose(filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv3DTranspose(filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv3D(6, (1, 1, 1), activation='softmax') (c9)

    return Model(inputs=[inputs], outputs=[outputs])







