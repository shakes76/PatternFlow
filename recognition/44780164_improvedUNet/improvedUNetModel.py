import tensorflow as tf

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
CONV_ARGUMENTS = dict(padding='same', kernel_initializer='he_uniform',
                      kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
leakyReLU = tf.keras.layers.LeakyReLU(alpha=0.01)


def dice_similarity_coefficient(prediction, actual):
    """
    The Dice Similarity Coefficient of the real mask and the predicted mask.
    Parameters
    ----------
    prediction : the predicted mask
    actual : the actual mask

    Returns
    -------
    float : The dice similarity coefficient of the two masks
    """
    x = tf.keras.backend.flatten(actual)
    y = tf.keras.backend.flatten(prediction)
    return (2.0 * tf.math.reduce_sum(x * y)) / (tf.math.reduce_sum(x) + tf.math.reduce_sum(y))


def dice_coe_loss(prediction, actual):
    """

    Parameters
    ----------
    prediction
    actual

    Returns
    -------
    float : the dice similarity coeffiient loss
    """
    return 1 - dice_similarity_coefficient(prediction, actual)


def improvedUNet():
    """
    An Improved UNET based on https://arxiv.org/pdf/1802.10508v1.pdf

    All values were taken from the paper.
    Returns
    -------
    tf.keras.Model : The improved UNet model
    """
    input_shape = tf.keras.Input((256, 256, 1))

    # Going Down
    connector1 = tf.keras.layers.Conv2D(16, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(input_shape)

    context1 = tf.keras.layers.Conv2D(16, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(connector1)
    context1 = tf.keras.layers.Dropout(0.3)(context1)
    context1 = tf.keras.layers.Conv2D(16, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(context1)

    layer1 = tf.keras.layers.Add()([connector1, context1])

    connector2 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation=leakyReLU, **CONV_ARGUMENTS)(layer1)
    context2 = tf.keras.layers.Conv2D(32, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(connector2)
    context2 = tf.keras.layers.Dropout(0.3)(context2)
    context2 = tf.keras.layers.Conv2D(32, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(context2)

    layer2 = tf.keras.layers.Add()([connector2, context2])

    connector3 = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation=leakyReLU, **CONV_ARGUMENTS)(layer2)
    context3 = tf.keras.layers.Conv2D(64, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(connector3)
    context3 = tf.keras.layers.Dropout(0.3)(context3)
    context3 = tf.keras.layers.Conv2D(64, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(context3)

    layer3 = tf.keras.layers.Add()([connector3, context3])

    connector4 = tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation=leakyReLU, **CONV_ARGUMENTS)(
        layer3)
    context4 = tf.keras.layers.Conv2D(128, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(connector4)
    context4 = tf.keras.layers.Dropout(0.3)(context4)
    context4 = tf.keras.layers.Conv2D(128, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(context4)

    layer4 = tf.keras.layers.Add()([connector4, context4])

    # (Stuck in) The Middle (With You)
    connector5_left = tf.keras.layers.Conv2D(256, (3, 3), strides=2, activation=leakyReLU, **CONV_ARGUMENTS)(
        layer4)
    context5 = tf.keras.layers.Conv2D(256, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(connector5_left)
    context5 = tf.keras.layers.Dropout(0.3)(context5)
    context5 = tf.keras.layers.Conv2D(256, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(context5)

    layer5 = tf.keras.layers.Add()([connector5_left, context5])

    connector5_right = tf.keras.layers.UpSampling2D(size=(2, 2))(layer5)
    connector5_right = tf.keras.layers.Conv2D(128, (2, 2), activation=leakyReLU, **CONV_ARGUMENTS)(connector5_right)
    connector5_right = tf.concat([layer4, connector5_right], axis=3)

    # Going Up
    context6 = tf.keras.layers.Conv2D(128, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(connector5_right)
    context6 = tf.keras.layers.Conv2D(128, (1, 1), activation=leakyReLU, **CONV_ARGUMENTS)(context6)
    connector6 = tf.keras.layers.UpSampling2D(size=(2, 2))(context6)
    connector6 = tf.keras.layers.Conv2D(64, (2, 2), activation=leakyReLU, **CONV_ARGUMENTS)(connector6)

    layer6 = tf.concat([layer3, connector6], axis=3)

    context7 = tf.keras.layers.Conv2D(64, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(layer6)
    context7 = tf.keras.layers.Conv2D(64, (1, 1), activation=leakyReLU, **CONV_ARGUMENTS)(context7)
    connector7 = tf.keras.layers.UpSampling2D(size=(2, 2))(context7)
    connector7 = tf.keras.layers.Conv2D(32, (2, 2), activation=leakyReLU, **CONV_ARGUMENTS)(connector7)

    layer7 = tf.concat([layer2, connector7], axis=3)

    context8 = tf.keras.layers.Conv2D(32, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(layer7)
    context8 = tf.keras.layers.Conv2D(32, (1, 1), activation=leakyReLU, **CONV_ARGUMENTS)(context8)
    connector8 = tf.keras.layers.UpSampling2D(size=(2, 2))(context8)
    connector8 = tf.keras.layers.Conv2D(16, (2, 2), activation=leakyReLU, **CONV_ARGUMENTS)(connector8)

    layer8 = tf.concat([layer1, connector8], axis=3)

    # Segmentation Layer
    segment1 = tf.keras.layers.Conv2D(32, (3, 3), activation=leakyReLU, **CONV_ARGUMENTS)(layer8)
    segment1 = tf.keras.layers.Conv2D(2, (1, 1), activation=leakyReLU, **CONV_ARGUMENTS)(segment1)

    segment2 = tf.keras.layers.Conv2D(2, (1, 1), activation=leakyReLU, **CONV_ARGUMENTS)(context8)

    segment3 = tf.keras.layers.Conv2D(2, (1, 1), activation=leakyReLU, **CONV_ARGUMENTS)(context7)

    segment3_connector = tf.keras.layers.UpSampling2D(size=(2, 2))(segment3)
    segment2 = tf.keras.layers.Add()([segment3_connector, segment2])
    segment2 = tf.keras.layers.UpSampling2D(size=(2, 2))(segment2)

    segment_out = tf.keras.layers.Add()([segment1, segment2])

    # Output
    final = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax')(segment_out)
    model = tf.keras.Model(inputs=input_shape, outputs=final)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=dice_coe_loss,
                  metrics=['accuracy', dice_similarity_coefficient])
    return model
