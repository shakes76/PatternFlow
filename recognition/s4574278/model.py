from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import random_normal
from tensorflow.python.keras.backend import l2_normalize
from enum import Enum

"""
Assume input width/height to be 416/480/512 (could be any the multiple of 32)
"""

##########################################################
#   CONSTANTS
##########################################################


class ModelSize(Enum):
    Tiny = (0.33, 0.375)
    S = (0.33, 0.50)
    M = (0.67, 0.75)
    L = (1.00, 1.00)
    X = (1.33, 1.25)


##########################################################
#   Modified CSPDarkNet53
#   Input -> Focus -> stem
#   -> CSP Layer * 4 (last layer go through SPP bottleneck)
#   -> extract last 3 layers of "Feature Pyramid"
#   -> Fusion (not part of backbone network)
##########################################################


def cspnet(input, size: ModelSize):
    """Generate a CSPNet with a corresponding size given by trainer.

    Args:
        x (Tensor): input
        size (ModelSize): controls the depth & width of the model
    """
    depth, width = size.value
    baseline = 64
    base_channels = int(width * baseline)  # 64
    base_depth = max(round(3 * depth), 1)  # 3
    # stem 512,512,3 => 256,256,64
    x = focus_layer(in_channels=base_channels, kernel_size=3, name="backbone/stem")(
        input
    )
    # dark2: 256,256,64 => 128,128,128
    x = res_block(base_channels, base_channels * 2, name="backbone/dark2")(x)
    # dark2 = x # we don't need the result of first res_block

    # dark 3: 128,128,128 => 64,64,256
    x = res_block(
        base_channels * 2,
        base_channels * 4,
        depth=base_depth * 3,
        name="backbone/dark3",
    )(x)
    dark3 = x
    # dark4: 64,64,256 => 32,32,512
    x = res_block(
        base_channels * 4,
        base_channels * 8,
        depth=base_depth * 3,
        name="backbone/dark4",
    )(x)
    dark4 = x

    # dark 5: 32,32,512 => 16,16,1024
    x = last_res_block(
        base_channels * 8, base_channels * 16, depth=base_depth, name="backbone/dark5"
    )(x)
    dark5 = x
    return (dark3, dark4, dark5)


##########################################################
#   Network blocks
##########################################################
def focus_layer(in_channels, kernel_size=3, name="stem"):
    """Lossless interlaced down-sampling:
    Quadruple channels & Half the width and the height"""

    def func(x):
        x = tf.concat(
            [
                x[..., ::2, ::2, :],
                x[..., 1::2, ::2, :],
                x[..., ::2, 1::2, :],
                x[..., 1::2, 1::2, :],
            ],
            axis=-1,
        )
        return conv_block(in_channels, kernel_size, name=name)(x)

    return func


def res_block(in_channels, out_channels, depth):
    return keras.Sequential(
        [
            conv_block(in_channels, out_channels, kernel_size=3, strides=2),
            csp_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                num_bottleneck=depth,
            ),
        ]
    )


def csp_layer(in_channels, out_channels, num_bottleneck=1, expansion=0.5):
    def func(x):
        hidden_channels = int(out_channels * expansion)
        conv1 = conv_block(in_channels, hidden_channels, kernel_size=1, stride=1)
        conv2 = conv_block(in_channels, hidden_channels, kernel_size=1, stride=1)
        conv3 = conv_block(2 * hidden_channels, out_channels, kernel_size=1, stride=1)
        modules = keras.Sequential(
            *[
                bottleneck(
                    hidden_channels,
                    hidden_channels,
                )
                for _ in range(num_bottleneck)
            ]
        )
        x_1 = conv1(x)
        x_2 = conv2(x)
        x_1 = modules(x_1)

        x = tf.cat((x_1, x_2), dim=1)
        return conv3(x)

    return func


def bottleneck(in_channels, out_channels, shortcut=True, expansion=0.5):
    def func(x):
        hidden_channels = int(out_channels * expansion)
        conv1 = conv_block(in_channels, hidden_channels, kernel_size=1, stride=1)
        conv2 = conv_block(hidden_channels, out_channels, 3, stride=1)
        y = conv2(conv1(x))
        if shortcut and in_channels == out_channels:
            y = y + x
        return y

    return func


def last_res_block(in_channels, out_channels, depth, name="backbone/dark5"):
    def func(x):
        x = conv_block(
            in_channels, out_channels, kernel_size=3, strides=2, name="{name}/conv"
        )(x)
        x = spp_bottleneck(in_channels=out_channels, out_channels=out_channels)(x)
        x = csp_layer(
            in_channels=out_channels,
            out_channels=out_channels,
            num_bottleneck=depth,
        )(x)
        return x

    return func


def spp_bottleneck(in_channels, out_channels, kernel_size=(5, 9, 13), name=""):
    pass


##########################################################
#   General Building blocks
##########################################################
def conv_block(
    in_channels, out_channels, kernel_size=1, strides=1, activation=tf.nn.silu, name=""
):
    """Conv2D -> BNorm -> SiLU"""
    padding = "valid" if strides == 2 else "same"
    return keras.Sequential(
        [
            layers.Conv2D(
                in_channels,
                out_channels,
                kernel_initializer=random_normal(stddev=0.2),
                kernel_regularizer=l2_normalize(5e-4),
                padding=padding,
                name=f"{name}/conv{kernel_size}",
                use_bias=False,
            ),
            layers.BatchNormalization(out_channels, name=f"{name}/bn"),
            layers.Activation(activation, name=f"{name}/silu"),
        ]
    )


##########################################################
# Helpers
##########################################################


def bbox_to_center(boxes: List):
    """convert (top-left,bottom-right) bbox to (center-x, center-y, width, height)"""
    pass


def bbox_to_corner(boxes: List):
    """convert (center-x, center-y, width, height) bbox to (top-left, bottom-right)"""
    pass
