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
    # decoder
    up_1_up = UpSampling3D(size=(2, 2, 2))(add_5)
    up_1_conv = Conv3D(filters_1 * 8, (3, 3, 3), padding="same", activation=LeakyRelu)(up_1_up)
    concat_1 = Concatenate()([up_1_conv, add_4])
    local_1_f = Conv3D(filters_1 * 8, (3, 3, 3), padding="same", activation=LeakyRelu)(concat_1)
    local_1_b = Conv3D(filters_1 * 8, (1, 1, 1), padding="same", activation=LeakyRelu)(local_1_f)
    up_2_up = UpSampling3D(size=(2, 2, 2))(local_1_b)
    up_2_conv = Conv3D(filters_1 * 4, (3, 3, 3), padding="same", activation=LeakyRelu)(up_2_up)
    concat_2 = Concatenate()([up_2_conv, add_3])
    local_2_f = Conv3D(filters_1 * 4, (3, 3, 3), padding="same", activation=LeakyRelu)(concat_2)
    local_2_b = Conv3D(filters_1 * 4, (1, 1, 1), padding="same", activation=LeakyRelu)(local_2_f)
    # get first part of output
    output_1 = UpSampling3D(size=(2, 2, 2))(local_2_b)
    output_1 = Conv3D(filters_1 * 2, (1, 1, 1), padding="same", activation=LeakyRelu)(output_1)
    up_3_up = UpSampling3D(size=(2, 2, 2))(local_2_b)
    up_3_conv = Conv3D(filters_1 * 2, (3, 3, 3), padding="same", activation=LeakyRelu)(up_3_up)
    concat_3 = Concatenate()([up_3_conv, add_2])
    local_3_f = Conv3D(filters_1 * 2, (3, 3, 3), padding="same", activation=LeakyRelu)(concat_3)
    local_3_b = Conv3D(filters_1 * 2, (1, 1, 1), padding="same", activation=LeakyRelu)(local_3_f)
    # get second part of output and combine them together
    output_2 = Add()([local_3_b, output_1])
    output_2 = UpSampling3D(size=(2, 2, 2))(output_2)
    up_4_up = UpSampling3D(size=(2, 2, 2))(local_3_b)
    up_4_conv = Conv3D(filters_1 * 1, (3, 3, 3), padding="same", activation=LeakyRelu)(up_4_up)
    concat_4 = Concatenate()([up_4_conv, add_1])
    conv_end = Conv3D(filters_1 * 2, (3, 3, 3), padding="same", activation=LeakyRelu)(concat_4)
    # get last part of output and combine them together
    output_3 = Add()([conv_end, output_2])
    # get final output with 2 categories, indicate whether the voxel is black or white.
    output = Dense(2, activation="softmax")(output_3)
    model = Model(inputs, output)
    print('Model build success')
    return model
