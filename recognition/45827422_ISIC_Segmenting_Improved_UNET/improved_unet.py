"""
     Author : Aravind Punugu
 Student ID : 45827422
       Date : 28 October 2021
GitHub Name : Tannishpage
"""

import tensorflow as tf
from tensorflow.keras import layers as l

def dice_similarity(real, pred):
    """
    Straightforward implementation of the DSC formula from wikipedia
    """
    real_flattened = tf.keras.backend.flatten(real)
    pred_flattened = tf.keras.backend.flatten(pred)
    numerator = 2 * (tf.keras.backend.sum(real_flattened*pred_flattened))
    denominator = tf.keras.backend.sum(real_flattened) + tf.keras.backend.sum(pred_flattened)

    return numerator/denominator


def create_model(img_size):
    """
    Creates the improved UNET model as described in:
    F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein,
    “Brain Tumor Segmentation and Radiomics Survival Prediction:
        Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online].
    Available: https://arxiv.org/abs/1802.10508v1
    """

    inputl = l.Input(img_size)

    # Downsampling
    conv1 = l.Conv2D(16, (3, 3), strides=1, padding='same')(inputl)
    conv1 = l.LeakyReLU()(conv1)

    ctx1 = l.Conv2D(16, (3, 3), strides=1, padding='same')(conv1)
    ctx1 = l.LeakyReLU()(ctx1)
    ctx1 = l.Conv2D(16, (3, 3), strides=1, padding='same')(ctx1)
    ctx1 = l.LeakyReLU()(ctx1)
    ctx1 = l.Dropout(0.3)(ctx1)
    ctx1 = l.add([conv1, ctx1])

    conv2 = l.Conv2D(32, (3, 3), strides=2, padding='same')(ctx1)
    conv2 = l.LeakyReLU()(conv2)


    ctx2 = l.Conv2D(32, (3, 3), strides=1, padding='same')(conv2)
    ctx2 = l.LeakyReLU()(ctx2)
    ctx2 = l.Conv2D(32, (3, 3), strides=1, padding='same')(ctx2)
    ctx2 = l.LeakyReLU()(ctx2)
    ctx2 = l.Dropout(0.3)(ctx2)
    ctx2 = l.add([conv2, ctx2])


    conv3 = l.Conv2D(64, (3, 3), strides=2, padding='same')(ctx2)
    conv3 = l.LeakyReLU()(conv3)

    ctx3 = l.Conv2D(64, (3, 3), strides=1, padding='same')(conv3)
    ctx3 = l.LeakyReLU()(ctx3)
    ctx3 = l.Conv2D(64, (3, 3), strides=1, padding='same')(ctx3)
    ctx3 = l.LeakyReLU()(ctx3)
    ctx3 = l.Dropout(0.3)(ctx3)
    ctx3 = l.add([conv3, ctx3])


    conv4 = l.Conv2D(128, (3, 3), strides=2, padding='same')(ctx3)
    conv4 = l.LeakyReLU()(conv4)

    ctx4 = l.Conv2D(128, (3, 3), strides=1, padding='same')(conv4)
    ctx4 = l.LeakyReLU()(ctx4)
    ctx4 = l.Conv2D(128, (3, 3), strides=1, padding='same')(ctx4)
    ctx4 = l.LeakyReLU()(ctx4)
    ctx4 = l.Dropout(0.3)(ctx4)
    ctx4 = l.add([conv4, ctx4])


    conv5 = l.Conv2D(256, (3, 3), strides=2, padding='same')(ctx4)
    conv5 = l.LeakyReLU()(conv5)

    ctx5 = l.Conv2D(256, (3, 3), strides=1, padding='same')(conv5)
    ctx5 = l.LeakyReLU()(ctx5)
    ctx5 = l.Conv2D(256, (3, 3), strides=1, padding='same')(ctx5)
    ctx5 = l.LeakyReLU()(ctx5)
    ctx5 = l.Dropout(0.3)(ctx5)
    ctx5 = l.add([conv5, ctx5])

    # Upsampling

    upsample1 = l.UpSampling2D((2, 2))(ctx5)
    upsample1 = l.Conv2D(128, (3, 3), strides=1, padding='same')(upsample1)
    upsample1 = l.LeakyReLU()(upsample1)
    upsample1 = l.concatenate([upsample1, ctx4])

    local1 = l.Conv2D(128, (3, 3), strides=1, padding='same')(upsample1)
    local1 = l.LeakyReLU()(local1)
    local1 = l.Conv2D(128, (1, 1), strides=1, padding='same')(local1)
    local1 = l.LeakyReLU()(local1)


    upsample2 = l.UpSampling2D((2, 2))(local1)
    upsample2 = l.Conv2D(64, (3, 3), strides=1, padding='same')(upsample2)
    upsample2 = l.LeakyReLU()(upsample2)
    upsample2 = l.concatenate([upsample2, ctx3])

    local2 = l.Conv2D(64, (3, 3), strides=1, padding='same')(upsample2)
    local2 = l.LeakyReLU()(local2)
    local2 = l.Conv2D(64, (1, 1), strides=1, padding='same')(local2)
    local2 = l.LeakyReLU()(local2)

    segment1 = l.Conv2D(16, (1, 1), strides=1, padding='same')(local2)
    segment1 = l.LeakyReLU()(segment1)
    segment1 = l.UpSampling2D((2, 2))(segment1)

    upsample3 = l.UpSampling2D((2, 2))(local2)
    upsample3 = l.Conv2D(32, (3, 3), strides=1, padding='same')(upsample3)
    upsample3 = l.LeakyReLU()(upsample3)
    upsample3 = l.concatenate([upsample3, ctx2])

    local3 = l.Conv2D(32, (3, 3), strides=1, padding='same')(upsample3)
    local3 = l.LeakyReLU()(local3)
    local3 = l.Conv2D(32, (1, 1), strides=1, padding='same')(local3)
    local3 = l.LeakyReLU()(local3)

    segment2 = l.Conv2D(16, (1, 1), strides=1, padding='same')(local3)
    segment2 = l.LeakyReLU()(segment2)
    segment2 = l.add([segment1, segment2])
    segment2 = l.UpSampling2D((2, 2))(segment2)

    upsample4 = l.UpSampling2D((2, 2))(local3)
    upsample4 = l.Conv2D(16, (3, 3), strides=1, padding='same')(upsample4)
    upsample4 = l.LeakyReLU()(upsample4)
    upsample4 = l.concatenate([upsample4, ctx1])

    last_conv = l.Conv2D(32, (3, 3), strides=1, padding='same')(upsample4)

    segment3 = l.Conv2D(16, (1, 1), strides=1, padding='same')(last_conv)
    segment3 = l.LeakyReLU()(segment3)
    segment3 = l.add([segment2, segment3])

    output = l.Conv2D(2, (1, 1), padding='same', activation='softmax')(segment3)

    model = tf.keras.Model(inputs=inputl, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='binary_crossentropy',
                    metrics=[dice_similarity])

    return model
