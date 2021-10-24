from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, Input, concatenate, UpSampling2D


def improved_unet(height, width, channels):
    fil = 16
    kern = (3, 3)
    act = 'relu'
    kern_init = 'he_normal'
    pad = 'same'
    stride = 2
    drop = 0.3
    # Building the model
    inputs = Input((height, width, channels))

    # Contraction

    # 3x3x3 convolution - 16 filters
    conv1 = Conv2D(fil, kern, activation=act, kernel_initializer=kern_init, padding=pad)(inputs)
    # Context module
    context1 = Conv2D(fil, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(conv1)
    context1 = Dropout(drop)(context1)
    context1 = Conv2D(fil, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(context1)
    # Concatenation
    concat1 = concatenate([conv1, context1])

    # 3x3x3 stride 2 convolution - 32 filters
    conv2 = Conv2D(fil * 2, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(concat1)
    # Context module
    context2 = Conv2D(fil * 2, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(conv2)
    context2 = Dropout(drop)(context2)
    context2 = Conv2D(fil * 2, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(
        context2)
    # Concatenation
    concat2 = concatenate([conv2, context2])

    # 3x3x3 stride 2 convolution - 64 filters
    conv3 = Conv2D(fil * 4, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(concat2)
    # Context module
    context3 = Conv2D(fil * 4, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(conv3)
    context3 = Dropout(drop)(context3)
    context3 = Conv2D(fil * 4, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(
        context3)
    # Concatenation
    concat3 = concatenate([conv3, context3])

    # 3x3x3 stride 2 convolution - 128 filters
    conv4 = Conv2D(fil * 8, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(concat3)
    # Context module
    context4 = Conv2D(fil * 8, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(conv4)
    context4 = Dropout(drop)(context4)
    context4 = Conv2D(fil * 8, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(
        context4)
    # Concatenation
    concat4 = concatenate([conv4, context4])

    # 3x3x3 stride 2 convolution - 256 filters
    conv5 = Conv2D(fil * 16, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(concat4)
    # Context module
    context5 = Conv2D(fil * 16, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(conv5)
    context5 = Dropout(drop)(context5)
    context5 = Conv2D(fil * 16, kern, activation=act, kernel_initializer=kern_init, padding=pad, strides=stride)(
        context5)
    # Concatenation
    concat5 = concatenate([conv5, context5])
    # Up-sampling - 128 filters
    upsamp1 = Conv2DTranspose(fil * 8, kern, activation=act, kernel_initializer=kern_init,
                              padding=pad, strides=stride)(concat5)  # CHANGE THIS

    # Expansive path

    # Concatenation
    concat6 = concatenate([upsamp1, concat4])
    # Localisation - 128 filters
    local1 = Conv2D(fil * 8, kern, activation=act, kernel_initializer=kern_init, padding=pad)(concat6)
    local1 = Conv2D(fil * 8, (1, 1), activation=act, kernel_initializer=kern_init, padding=pad)(local1)
    # Up-sampling - 64 filters
    upsamp2 = Conv2DTranspose(fil * 4, kern, activation=act, kernel_initializer=kern_init,
                              padding=pad, strides=stride)(local1)

    # Concatenation
    concat7 = concatenate([upsamp2, concat3])
    # Localisation - 64 filters
    local2 = Conv2D(fil * 4, kern, activation=act, kernel_initializer=kern_init, padding=pad)(concat7)
    local2 = Conv2D(fil * 4, (1, 1), activation=act, kernel_initializer=kern_init, padding=pad)(local2)
    # Segmentation
    segment1 = local2
    # Up-sampling - 32 filters
    upsamp3 = Conv2DTranspose(fil * 2, kern, activation=act, kernel_initializer=kern_init,
                              padding=pad, strides=stride)(local2)

    # Concatenation
    concat8 = concatenate([upsamp3, concat2])
    # Localisation - 32 filters
    local3 = Conv2D(fil * 2, kern, activation=act, kernel_initializer=kern_init, padding=pad)(concat8)
    local3 = Conv2D(fil * 2, (1, 1), activation=act, kernel_initializer=kern_init, padding=pad)(local3)
    # Segmentation
    segment2 = local3
    # Up-sampling - 16 filters
    upsamp4 = Conv2DTranspose(fil, kern, activation=act, kernel_initializer=kern_init,
                              padding=pad, strides=stride)(local3)

    # Concatenation
    concat9 = concatenate([upsamp4, concat1])
    # 3x3x3 convolution - 32 filters
    conv6 = Conv2D(fil, kern, activation=act, kernel_initializer=kern_init, padding=pad)(concat9)
    # Segmentation
    segment3 = conv6

    outputs = Conv2D(1, (1, 1), activation='softmax')(conv6)
    return inputs, outputs
