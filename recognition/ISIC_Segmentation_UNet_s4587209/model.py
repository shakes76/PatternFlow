import tensorflow as tf
from tensorflow.keras.layers import concatenate, Conv2D, BatchNormalization, Dropout, Input, LeakyReLU, UpSampling2D


def improved_unet(height, width, channels):
    fil = 16
    kern = (3, 3)
    pad = 'same'
    drop = 0.3
    alp = 0.01
    # Building the model
    inputs = Input((height, width, channels))

    # 3x3x3 convolution - 16 filters
    conv1 = Conv2D(fil, kern, padding=pad)(inputs)
    # Instance Normalisation
    bat1 = BatchNormalization()(conv1)
    # Leaky ReLU
    relu1 = LeakyReLU(alpha=alp)(bat1)
    # Context module
    context1 = Conv2D(fil, kern, padding=pad)(relu1)
    context1 = Dropout(drop)(context1)
    context1 = Conv2D(fil, kern, padding=pad)(context1)
    # Instance Normalisation
    bat1 = BatchNormalization()(context1)
    # Leaky ReLU
    relu1 = LeakyReLU(alpha=alp)(bat1)
    # Element-wise sum
    out1 = conv1 + relu1

    # 3x3x3 stride 2 convolution - 32 filters
    conv2 = Conv2D(fil * 2, kern, padding=pad)(out1)
    # Instance Normalisation
    bat2 = BatchNormalization()(conv2)
    # Leaky ReLU
    relu2 = LeakyReLU(alpha=alp)(bat2)
    # Context module
    context2 = Conv2D(fil * 2, kern, padding=pad)(relu2)
    context2 = Dropout(drop)(context2)
    context2 = Conv2D(fil * 2, kern, padding=pad)(context2)
    # Instance Normalisation
    bat2 = BatchNormalization()(context2)
    # Leaky ReLU
    relu2 = LeakyReLU(alpha=alp)(bat2)
    # Element-wise sum
    out2 = conv2 + relu2

    # 3x3x3 stride 2 convolution - 64 filters
    conv3 = Conv2D(fil * 4, kern, padding=pad)(out2)
    # Instance Normalisation
    bat3 = BatchNormalization()(conv3)
    # Leaky ReLU
    relu3 = LeakyReLU(alpha=alp)(bat3)
    # Context module
    context3 = Conv2D(fil * 4, kern, padding=pad)(relu3)
    context3 = Dropout(drop)(context3)
    context3 = Conv2D(fil * 4, kern, padding=pad)(context3)
    # Instance Normalisation
    bat3 = BatchNormalization()(context3)
    # Leaky ReLU
    relu3 = LeakyReLU(alpha=alp)(bat3)
    # Element-wise sum
    out3 = conv3 + relu3

    # 3x3x3 stride 2 convolution - 128 filters
    conv4 = Conv2D(fil * 8, kern, padding=pad)(out3)
    # Instance Normalisation
    bat4 = BatchNormalization()(conv4)
    # Leaky ReLU
    relu4 = LeakyReLU(alpha=alp)(bat4)
    # Context module
    context4 = Conv2D(fil * 8, kern, padding=pad)(relu4)
    context4 = Dropout(drop)(context4)
    context4 = Conv2D(fil * 8, kern, padding=pad)(context4)
    # Instance Normalisation
    bat4 = BatchNormalization()(context4)
    # Leaky ReLU
    relu4 = LeakyReLU(alpha=alp)(bat4)
    # Element-wise sum
    out4 = conv4 + relu4

    # 3x3x3 stride 2 convolution - 256 filters
    conv5 = Conv2D(fil * 16, kern, padding=pad)(out4)
    # Instance Normalisation
    bat5 = BatchNormalization()(conv5)
    # Leaky ReLU
    relu5 = LeakyReLU(alpha=alp)(bat5)
    # Context module
    context5 = Conv2D(fil * 16, kern, padding=pad)(relu5)
    context5 = Dropout(drop)(context5)
    context5 = Conv2D(fil * 16, kern, padding=pad)(context5)
    # Instance Normalisation
    bat5 = BatchNormalization()(context5)
    # Leaky ReLU
    relu5 = LeakyReLU(alpha=alp)(bat5)
    # Element-wise sum
    out5 = conv5 + relu5
    # Up-sampling - 128 filters
    upsamp5 = UpSampling2D(size=(2, 2))(out5)
    upsamp5 = Conv2D(fil * 8, kern, padding=pad, strides=(2, 2))(upsamp5)

    # Expansive path

    # Instance Normalisation
    bat6 = BatchNormalization()(upsamp5)
    # Leaky ReLU
    relu6 = LeakyReLU(alpha=alp)(bat6)
    # Concatenation
    concat6 = concatenate([relu6, out4])
    # Localisation - 128 filters
    local6 = Conv2D(fil * 8, kern, padding=pad)(concat6)
    local6 = Conv2D(fil * 8, (1, 1), padding=pad)(local6)
    # Instance Normalisation
    bat6 = BatchNormalization()(local6)
    # Leaky ReLU
    relu6 = LeakyReLU(alpha=alp)(bat6)
    # Up-sampling - 64 filters
    upsamp6 = UpSampling2D(size=(2, 2))(relu6)
    upsamp6 = Conv2D(fil * 4, (2, 2), padding=pad, strides=(2, 2))(upsamp6)

    # Instance Normalisation
    bat7 = BatchNormalization()(upsamp6)
    # Leaky ReLU
    relu7 = LeakyReLU(alpha=alp)(bat7)
    # Concatenation
    concat7 = concatenate([relu7, out3])
    # Localisation - 64 filters
    local7 = Conv2D(fil * 4, kern, padding=pad)(concat7)
    local7 = Conv2D(fil * 4, (1, 1), padding=pad)(local7)
    # Instance Normalisation
    bat7 = BatchNormalization()(local7)
    # Leaky ReLU
    relu7 = LeakyReLU(alpha=alp)(bat7)
    # Segmentation
    segment7 = relu7
    # Up-sampling - 32 filters
    upsamp7 = UpSampling2D(size=(2, 2))(relu7)
    upsamp7 = Conv2D(fil * 2, (2, 2), padding=pad, strides=(2, 2))(upsamp7)

    # Instance Normalisation
    bat8 = BatchNormalization()(upsamp7)
    # Leaky ReLU
    relu8 = LeakyReLU(alpha=alp)(bat8)
    # Concatenation
    concat8 = concatenate([relu8, out2])
    # Localisation - 32 filters
    local8 = Conv2D(fil * 2, kern, padding=pad)(concat8)
    local8 = Conv2D(fil * 2, (1, 1), padding=pad)(local8)
    # Instance Normalisation
    bat8 = BatchNormalization()(local8)
    # Leaky ReLU
    relu8 = LeakyReLU(alpha=alp)(bat8)
    # Segmentation
    segment8 = relu8
    # Up-sampling - 16 filters
    upsamp8 = UpSampling2D(size=(2, 2))(relu8)
    upsamp8 = Conv2D(fil, (2, 2), padding=pad, strides=(2, 2))(upsamp8)

    # Instance Normalisation
    bat9 = BatchNormalization()(upsamp8)
    # Leaky ReLU
    relu9 = LeakyReLU(alpha=alp)(bat9)
    # Concatenation
    concat9 = concatenate([relu9, out1])
    # 3x3x3 convolution - 32 filters
    conv9 = Conv2D(fil, kern, padding=pad)(concat9)
    # Instance Normalisation
    bat9 = BatchNormalization()(conv9)
    # Leaky ReLU
    relu9 = LeakyReLU(alpha=alp)(bat9)
    # Segmentation
    segment9 = relu9

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
