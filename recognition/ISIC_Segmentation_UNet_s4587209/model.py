import tensorflow as tf
from tensorflow.keras.layers import concatenate, Conv2D, Dropout, Input, UpSampling2D


def improved_unet(height, width, channels):
    fil = 16
    kern = (3, 3)
    
    pad = 'same'
    drop = 0.3
    # Building the model
    inputs = Input((height, width, channels))

    # 3x3x3 convolution - 16 filters
    conv1 = Conv2D(fil, kern, padding=pad)(inputs)
    # Context module
    context1 = Conv2D(fil, kern, padding=pad)(conv1)
    context1 = Dropout(drop)(context1)
    context1 = Conv2D(fil, kern, padding=pad)(context1)
    # Element-wise sum
    out1 = conv1 + context1

    # 3x3x3 stride 2 convolution - 32 filters
    conv2 = Conv2D(fil * 2, kern, padding=pad)(out1)
    # Context module
    context2 = Conv2D(fil * 2, kern, padding=pad)(conv2)
    context2 = Dropout(drop)(context2)
    context2 = Conv2D(fil * 2, kern, padding=pad)(
        context2)
    # Element-wise sum
    out2 = conv2 + context2

    # 3x3x3 stride 2 convolution - 64 filters
    conv3 = Conv2D(fil * 4, kern, padding=pad)(out2)
    # Context module
    context3 = Conv2D(fil * 4, kern, padding=pad)(conv3)
    context3 = Dropout(drop)(context3)
    context3 = Conv2D(fil * 4, kern, padding=pad)(
        context3)
    # Element-wise sum
    out3 = conv3 + context3

    # 3x3x3 stride 2 convolution - 128 filters
    conv4 = Conv2D(fil * 8, kern, padding=pad)(out3)
    # Context module
    context4 = Conv2D(fil * 8, kern, padding=pad)(conv4)
    context4 = Dropout(drop)(context4)
    context4 = Conv2D(fil * 8, kern, padding=pad)(
        context4)
    # Element-wise sum
    out4 = conv4 + context4

    # 3x3x3 stride 2 convolution - 256 filters
    conv5 = Conv2D(fil * 16, kern, padding=pad)(out4)
    # Context module
    context5 = Conv2D(fil * 16, kern, padding=pad)(conv5)
    context5 = Dropout(drop)(context5)
    context5 = Conv2D(fil * 16, kern, padding=pad)(
        context5)
    # Element-wise sum
    out5 = conv5 + context5
    # Up-sampling - 128 filters
    upsamp5 = UpSampling2D(size=(2, 2))(out5)
    upsamp5 = Conv2D(fil * 8, kern, padding=pad, strides=(2, 2))(upsamp5)

    # Expansive path

    # Concatenation
    concat6 = concatenate([upsamp5, out4])
    # Localisation - 128 filters
    local6 = Conv2D(fil * 8, kern, padding=pad)(concat6)
    local6 = Conv2D(fil * 8, (1, 1), padding=pad)(local6)
    # Up-sampling - 64 filters
    upsamp6 = UpSampling2D(size=(2, 2))(local6)
    upsamp6 = Conv2D(fil * 4, (2, 2), padding=pad, strides=(2, 2))(upsamp6)

    # Concatenation
    concat7 = concatenate([upsamp6, out3])
    # Localisation - 64 filters
    local7 = Conv2D(fil * 4, kern, padding=pad)(concat7)
    local7 = Conv2D(fil * 4, (1, 1), padding=pad)(local7)
    # Segmentation
    segment7 = local7
    # Up-sampling - 32 filters
    upsamp7 = UpSampling2D(size=(2, 2))(local7)
    upsamp7 = Conv2D(fil * 2, (2, 2), padding=pad, strides=(2, 2))(upsamp7)

    # Concatenation
    concat8 = concatenate([upsamp7, out2])
    # Localisation - 32 filters
    local8 = Conv2D(fil * 2, kern, padding=pad)(concat8)
    local8 = Conv2D(fil * 2, (1, 1), padding=pad)(local8)
    # Segmentation
    segment8 = local8
    # Up-sampling - 16 filters
    upsamp8 = UpSampling2D(size=(2, 2))(local8)
    upsamp8 = Conv2D(fil, (2, 2), padding=pad, strides=(2, 2))(upsamp8)

    # Concatenation
    concat9 = concatenate([upsamp8, out1])
    # 3x3x3 convolution - 32 filters
    conv9 = Conv2D(fil, kern, padding=pad)(concat9)
    # Segmentation
    segment9 = conv9

    # Upscale Segmented Layers and apply
    segment7 = UpSampling2D(size=(2, 2))(segment7)
    segment7 = Conv2D(fil * 2, (1, 1), strides=(2, 2))(segment7)
    segment8 = segment8 + segment7
    segment8 = UpSampling2D(size=(2, 2))(segment8)
    segment8 = Conv2D(fil, (1, 1), strides=(2, 2))(segment8)
    segment9 = segment9 + segment8

    # Softmax
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(segment9)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])
