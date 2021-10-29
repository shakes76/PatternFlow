import numpy as np
import tensorflow as tf
import nibabel
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, concatenate, Dropout, MaxPooling3D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from scipy import ndimage
import matplotlib.pyplot as plt

"""
Labels:
Background = 0
Body = 1
Bones = 2
Bladder = 3
Rectum = 4
Prostate = 5
"""


IMG_HEIGHT = 256
IMG_DEPTH = 256
IMG_WIDTH = 128
IMG_CHANNELS = 1


def get_nifti_data(file_name):
    img = nibabel.load(file_name).get_fdata()
    return img


def one_hot(file_name):
    img = get_nifti_data(file_name)
    encoding = np.zeros((IMG_HEIGHT, IMG_DEPTH, IMG_WIDTH, 6))
    for i, unique_value in enumerate(np.unique(img)):
        encoding[:, :, :, i][img == unique_value] = 1
    return encoding


def normalise(image):
    # subtract mean
    mean = np.average(image)
    image = image - mean

    # divide by sd
    sd = np.std(image)
    image = image / sd

    # unity-based normalisation
    # max_val = np.amax(image)
    # min_val = np.amin(image)
    # image = (image - min_val) / (max_val - min_val)
    return image


def trim(image, diff, axis):
    s_diff = diff // 2
    e_diff = s_diff + diff % 2
    for i in range(s_diff):
        image = np.delete(image, 0, axis=axis)
    for i in range(e_diff):
        image = np.delete(image, -1, axis=axis)
    return image


def reshape(dimension, image):
    return np.reshape(image, (IMG_HEIGHT, IMG_DEPTH, IMG_WIDTH, dimension))


def rotate(img, deg, is_mask):
    order = 3  # bspline interp
    if is_mask:
        order = 0  # NN interp

    img = ndimage.rotate(img, deg[0], reshape=False, prefilter=True, order=order)
    return img


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# def weighted_cross(beta):
#     def loss(y_true, y_pred):
#         weight_a = beta * tf.cast(y_true, tf.float32)
#         weight_b = 1 - tf.cast(y_true, tf.float32)
#
#         o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
#         return tf.reduce_mean(o)
#
#     return loss


def unet(filters):
    inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, IMG_CHANNELS))

    # Contraction
    c1 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    c5 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    # Expansion
    u6 = Conv3DTranspose(filters * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv3DTranspose(filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv3DTranspose(filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    # c9 = BatchNormalization()(c9)

    outputs = Conv3D(6, (1, 1, 1), activation='softmax')(c9)

    return Model(inputs=[inputs], outputs=[outputs])


def dice(y_test, y_pred, smooth=1):
    y_test_f = K.flatten(y_test)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_test_f * y_pred_f)
    d = (2. * intersect + smooth) / (K.sum(y_test_f) + K.sum(y_pred_f) + smooth)
    return d


def dice_loss(smooth=1):
    def dice_keras(y_true, y_pred):
        return 1 - dice(y_true, y_pred, smooth)
    return dice_keras


def plt_compare(img, test_mask, pred, num):
    # reshape
    img = np.reshape(img, (IMG_HEIGHT, IMG_DEPTH, IMG_WIDTH))
    test_mask = np.argmax(test_mask, axis=-1)
    pred = np.argmax(pred, axis=-1)

    # plot
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img[img.shape[0] // 2], cmap='gray')
    ax1.title.set_text("Image Slice")
    ax2.imshow(test_mask[test_mask.shape[0] // 2], cmap='gray')
    ax2.title.set_text("Test Mask")
    ax3.imshow(pred[pred.shape[0] // 2], cmap='gray')
    ax3.title.set_text("Prediction")
    fig1.savefig("pred_{}.png".format(num))
