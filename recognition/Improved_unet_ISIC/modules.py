import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Dense, UpSampling3D, Concatenate, Add
from tensorflow.keras.models import Model


# define some constants
filters_1 = 16
batch_size = 2
LeakyRelu = tf.keras.layers.LeakyReLU(alpha=0.01)


def create_model():
    """

    :return: Improved_unet created according to the essay and suitable for ISIC dataset
    """
    # encoder
    inputs = Input(shape=(64, 64, 16, 1))
    conv_start = Conv3D(filters_1, (3, 3, 3), padding="same", activation=LeakyRelu)(inputs)
    context_1_c1 = Conv3D(filters_1, (3, 3, 3), padding="same", activation=LeakyRelu)(conv_start)
    context_1_do = Dropout(0.3)(context_1_c1)
    context_1_c2 = Conv3D(filters_1, (3, 3, 3), padding="same", activation=LeakyRelu)(context_1_do)
    add_1 = Add()([conv_start, context_1_c2])
    conv_s2 = Conv3D(filters_1 * 2, (3, 3, 3), strides=(2, 2, 2), padding="same", activation=LeakyRelu)(add_1)
    context_2_c1 = Conv3D(filters_1 * 2, (3, 3, 3), padding="same", activation=LeakyRelu)(conv_s2)
    context_2_do = Dropout(0.3)(context_2_c1)
    context_2_c2 = Conv3D(filters_1 * 2, (3, 3, 3), padding="same", activation=LeakyRelu)(context_2_do)
    add_2 = Add()([conv_s2, context_2_c2])
    conv_s3 = Conv3D(filters_1 * 4, (3, 3, 3), strides=(2, 2, 2), padding="same", activation=LeakyRelu)(add_2)
    context_3_c1 = Conv3D(filters_1 * 4, (3, 3, 3), padding="same", activation=LeakyRelu)(conv_s3)
    context_3_do = Dropout(0.3)(context_3_c1)
    context_3_c2 = Conv3D(filters_1 * 4, (3, 3, 3), padding="same", activation=LeakyRelu)(context_3_do)
    add_3 = Add()([conv_s3, context_3_c2])
    conv_s4 = Conv3D(filters_1 * 8, (3, 3, 3), strides=(2, 2, 2), padding="same", activation=LeakyRelu)(add_3)
    context_4_c1 = Conv3D(filters_1 * 8, (3, 3, 3), padding="same", activation=LeakyRelu)(conv_s4)
    context_4_do = Dropout(0.3)(context_4_c1)
    context_4_c2 = Conv3D(filters_1 * 8, (3, 3, 3), padding="same", activation=LeakyRelu)(context_4_do)
    add_4 = Add()([conv_s4, context_4_c2])
    conv_s5 = Conv3D(filters_1 * 16, (3, 3, 3), strides=(2, 2, 2), padding="same", activation=LeakyRelu)(add_4)
    context_5_c1 = Conv3D(filters_1 * 16, (3, 3, 3), padding="same", activation=LeakyRelu)(conv_s5)
    context_5_do = Dropout(0.3)(context_5_c1)
    context_5_c2 = Conv3D(filters_1 * 16, (3, 3, 3), padding="same", activation=LeakyRelu)(context_5_do)
    add_5 = Add()([conv_s5, context_5_c2])
    pass
