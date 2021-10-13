import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, \
    Concatenate, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from typing import Tuple


def unet_conv2d(filters: int, name: str) -> Conv2D:
    return Conv2D(filters=filters, kernel_size=(3, 3), padding="valid",
                  activation=tf.keras.activations.relu,
                  name=name)


def unet_maxpool2d(name: str) -> MaxPool2D:
    return MaxPool2D(pool_size=(2, 2), strides=2, name=name)


def unet_upconv(filters: int, name: str) -> Conv2DTranspose:
    return Conv2DTranspose(filters=filters, kernel_size=(3, 3), name=name,
                           strides=(2, 2), padding="same")


def build_unet(width: int, height: int, levels: int,
               filters: int = 64) -> Model:
    inputs = Input(shape=(width, height, 1), name="input")
    _outputs = _build_unet(inputs, width, height, levels, filters=filters)
    outputs = Conv2D(filters=2, kernel_size=(1, 1),
                     padding="same",
                     activation=tf.keras.activations.relu,
                     name="output")(_outputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def _build_unet(inputs, width, height, levels, filters=64):
    conv_a = unet_conv2d(filters, "conv" + str(levels) + "a")(inputs)
    conv_b = unet_conv2d(filters, "conv" + str(levels) + "b")(conv_a)

    if levels == 1:
        return conv_b
    else:
        pool = unet_maxpool2d("pool" + str(levels))(conv_b)
        sub_unet = _build_unet(pool, (width - 4) // 2, (height - 4) // 2,
                               levels - 1, filters * 2)
        upconv = unet_upconv(filters, "upconv" + str(levels))(sub_unet)
        crop = Cropping2D(name="crop" + str(levels),
                          cropping=get_cropping(width, height, levels))(conv_b)
        concat = Concatenate(name="concat" + str(levels))([upconv, crop])
        conv_c = unet_conv2d(filters, "conv" + str(levels) + "c")(concat)
        conv_d = unet_conv2d(filters, "conv" + str(levels) + "d")(conv_c)
        return conv_d


def unet_end_length(length: int, level: int) -> int:
    if level == 1:
        return length - 4
    else:
        return unet_end_length((length - 4) // 2, level - 1) * 2 - 4


def get_cropping(width: int, height: int, levels: int) -> Tuple[int, int]:
    new_width = unet_end_length(width, levels) + 4
    new_height = unet_end_length(height, levels) + 4

    cropping_x = (width - 4 - new_width) // 2
    cropping_y = (height - 4 - new_height) // 2

    return cropping_x, cropping_y


if __name__ == "__main__":
    model = build_unet(572, 572, 5)
    model.summary()
