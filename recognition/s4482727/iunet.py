import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate,\
    Dropout, Add, UpSampling2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import Input

IMG_HEIGHT = 96
IMG_WIDTH = 128
IMG_CHANNELS = 3


def iunet_conv2d(filters: int, name: str) -> Conv2D:
    return Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
                  activation=tf.keras.activations.relu,
                  name=name)


def iunet_conv2d_stride2(filters: int, name: str) -> Conv2D:
    return Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
                  activation=tf.keras.activations.relu,
                  strides=2, name=name)


def iunet_context(inputs, filters, name):
    conv_a = iunet_conv2d(filters, name + "_conv_a")(inputs)
    conv_b = iunet_conv2d(filters, name + "_conv_b")(conv_a)
    dropout = Dropout(0.3, name=name + "_dropout")(conv_b)
    return dropout


def iunet_upsample(filters: int, name: str) -> Conv2DTranspose:
    return Conv2DTranspose(filters=filters, kernel_size=(3, 3),
                           name=name, strides=(2, 2),
                           padding="same")


def iunet_segment(name):
    return Conv2D(filters=1, kernel_size=(1, 1),
                  padding="same",
                  activation=None,
                  name=name)


def iunet_localize(inputs, filters, name):
    conv_a = iunet_conv2d(filters, name + "_conv_a")(inputs)
    conv_b = Conv2D(filters=filters, kernel_size=(1, 1), padding="same",
                    activation=tf.keras.activations.relu,
                    name=name + "conv_b")(conv_a)
    return conv_b


def build_iunet():
    inputs = Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        name="input")

    conv_1 = iunet_conv2d(16, "conv_1")(inputs)
    context_1 = iunet_context(conv_1, 16, "context_1")
    add_1 = Add(name="add_1")([conv_1, context_1])

    conv_2 = iunet_conv2d_stride2(32, "conv_2")(add_1)
    context_2 = iunet_context(conv_2, 32, "context_2")
    add_2 = Add(name="add_2")([conv_2, context_2])

    conv_3 = iunet_conv2d_stride2(64, "conv_3")(add_2)
    context_3 = iunet_context(conv_3, 64, "context_3")
    add_3 = Add(name="add_3")([conv_3, context_3])

    conv_4 = iunet_conv2d_stride2(128, "conv_4")(add_3)
    context_4 = iunet_context(conv_4, 128, "context_4")
    add_4 = Add(name="add_4")([conv_4, context_4])

    conv_5 = iunet_conv2d_stride2(256, "conv_5")(add_4)
    context_5 = iunet_context(conv_5, 256, "context_5")
    add_5 = Add(name="add_5")([conv_5, context_5])
    upsample_5 = iunet_upsample(128, "upsample_5")(add_5)

    concat_4 = Concatenate(name="concat_4")([add_4, upsample_5])
    localize_4 = iunet_localize(concat_4, 128, "localize_4")
    upsample_4 = iunet_upsample(64, "upsample_4")(localize_4)

    concat_3 = Concatenate(name="concat_3")([add_3, upsample_4])
    localize_3 = iunet_localize(concat_3, 64, "localize_3")
    upsample_3 = iunet_upsample(32, "upsample_3")(localize_3)
    segment_3 = iunet_segment("segment_3")(localize_3)

    concat_2 = Concatenate(name="concat_2")([add_2, upsample_3])
    localize_2 = iunet_localize(concat_2, 32, "localize_2")
    upsample_2 = iunet_upsample(16, "upsample_2")(localize_2)
    segment_2 = iunet_segment("segment_2")(localize_2)

    upscale_2 = UpSampling2D(size=(2,2), name="upscale_2")(segment_3)
    add_2b = Add(name="add_2b")([segment_2, upscale_2])

    concat_1 = Concatenate(name="concat_1")([add_1, upsample_2])
    conv_1b = iunet_conv2d(32, "conv_1b")(concat_1)
    segment_1 = iunet_segment("segment_1")(conv_1b)

    upscale_1 = UpSampling2D(size=(2, 2), name="upscale_1")(add_2b)
    add_1b = Add(name="add_1b")([segment_1, upscale_1])
    sigmoid_1 = Activation("sigmoid")(add_1b)

    model = Model(inputs=inputs, outputs=sigmoid_1)

    return model


if __name__ == "__main__":
    model = build_iunet()
    model.summary()
