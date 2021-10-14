import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, Concatinate, Input, LeakyReLU, SpatialDropout2D, InstanceNormalisation, Softmax
from tensorflow.keras.models import models, Sequential

# Reference: https://github.com/pykao/Modified-3D-UNet-Pytorch/blob/63f0489e8d1fdd7ec6a203bcff095f12ea030824/model.py#L70
class Unet():

    def context_module(input_layer, filters, stride):
        return Sequential([
            input_layer,
            LeakyReLU(),
            Conv2D(filters, 3, strides=stride, padding="same"),
            SpatialDropout2D(0.6),
            Conv2D(filters, 3, strides=stride, padding="same")
        ])


    def encoder_block(input_layer, filters, stride, switch=0):
        # Level 1 context pathway
        conv_layer = Conv2D(fitlers, 3, strides=stride, padding="same")(input_layer)
        residual = conv_layer
        # Context module
        c_module = LeakyReLU()(conv_layer)
        conv_layer = Conv2D(filters, 3, strides=stride, padding="same")(c_module)
        dropout = SpatialDropout2D(0.6)(conv_layer)
        c_module = Conv2D(filters, 3, strides=stride, padding="same")(dropout)
        # Element wise summation of convolution and context module
        c_module += residual
        # Save for skip connection
        context_1 = c_module
        norm_layer = InstanceNormalisation()(c_module)
        norm_layer = LeakyReLU()(norm_layer)
        context_2 = norm_layer
        if switch==0:
            return norm_layer, context_2
        else:
            return norm_layer, context_1

    def upsample_module(input_layer, filters):
        return Sequential([
            input_layer,
            InstanceNormalisation(),
            LeakyReLU(),
            UpSampling2D(2, interpolation="nearest"),
            Conv2D(filters, 3, stride=1, padding="same"),
            InstanceNormalisation(),
            LeakyReLU()
        ])

    def localisation_module(input_layer, context, filters):
        cat = Concatinate()([input_layer, context])
        conv_layer = Conv2D(filters, 3, stride=1, padding="same")(cat)
        norm = InstanceNormalisation()(conv_layer)
        norm = LeakyReLU()(norm)
        # Segmentation layer for deep supervision
        seg = norm
        return Conv2D(filters, 3, stride=1, padding="same")(norm), seg

    def build_model(input_size):
        # Level 1 context pathway
        input_layer = Input(input_size)
        out, context_1 = encoder_block(input_layer, 64, 1, switch=1)
        context_1 = LeakyReLU()(context_1)

        # Level 2 context pathway
        out, context_2 = encoder_block(out, 128, 2)

        # Level 3 context gateway
        out, context_3 = encoder_block(out, 256, 2)

        # Level 4 context gateway
        out, context_4 = encoder_block(out, 512, 2)

        # Level 5
        # Dump Normalisation
        dump, out = encoder_block(out, 1024, 2, switch=1)
        out = upsample_module(out, 512)
        out = Conv2D(512, 1, stride=1, padding="same")(out)
        out = InstanceNormalisation()(out)
        out = LeakyReLU()(out)

        # Level 1 localistaion pathway
        # Dump segmentation layer at level 1, we don't need it yet
        out, dump = localisation_module(out, context_4, 512)
        out = upsample_module(out, 256)

        # Level 2 localistaion pathway
        out, seg_1 = localisation_module(out, context_3, 256)
        out = upsample_module(out, 128)

        # Level 3 localisation pathway
        out, seg_2 = localisation_module(out, context_2, 128)
        out = upsample_module(out, 64)

        # Level 4 localisation pathway
        out, dump = localisation_moduleout, context_1, 64)

        # Deep supervision
        seg_1 = Conv2D(256, 3, stride=1, padding="same")(seg_1)
        seg_1 = UpSampling2D(2, interpolation="nearest")(seg_1)
        seg_2 = Conv2D(256, 3, stride=1, padding="same")(seg_2)
        seg_layer = seg_1 + seg_2
        seg_layer = UpSampling2D(2, interpolation="nearest")(seg_layer)
        out = out + seg_layer
        return Softmax()(out)
