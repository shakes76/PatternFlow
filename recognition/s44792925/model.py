import tensorflow as tf

def context_module(input, conv):
    """Context module architecture modelled from https://www.researchgate.net/figure/Architecture-of-normal-residual-block-a-and-pre-activation-residual-block-b_fig2_337691625"""

    block = tf.keras.layers.BatchNormalization()(input)
    block = tf.keras.layers.ReLU()(block)
    block = tf.keras.layers.Conv2D(conv, (3,3), padding='same')(block)
    
    block = tf.keras.layers.Dropout(0.3)(block)

    block = tf.keras.layers.BatchNormalization()(block)
    block = tf.keras.layers.ReLU()(block)
    block = tf.keras.layers.Conv2D(conv, (3,3), padding='same')(block)

    return block

def model(height, width, channel):
    """Model has been structured as per the architecture defined at https://arxiv.org/pdf/1802.10508v1.pdf"""

    input = tf.keras.layers.Input((height, width, channel))

    #encode data
    #block 1
    conv1 = tf.keras.layers.Conv2D(16, (3,3), padding='same')(input)
    context_module1 = context_module(conv1, 16)
    sum1 = tf.keras.layers.Add()([conv1, context_module1])
    #block 2
    conv2 = tf.keras.layers.Conv2D(32, (3,3), strides=(2,2), padding='same')(sum1)
    context_module2 = context_module(conv2, 32)
    sum2 = tf.keras.layers.Add()([conv2, context_module2])
    #block 3
    conv3 = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same')(sum2)
    context_module3 = context_module(conv3, 64)
    sum3 = tf.keras.layers.Add()([conv3, context_module3])
    #block 4
    conv4 = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(sum3)
    context_module4 = context_module(conv4, 128)
    sum4 = tf.keras.layers.Add()([conv4, context_module4])
    #block 5
    conv5 = tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same')(sum4)
    context_module5 = context_module(conv5, 256)
    sum5 = tf.keras.layers.Add()([conv5, context_module5])
    
    #recombine representations to localise notable features used in segmentation
    upsampling1 = tf.keras.layers.UpSampling2D(size=(2,2))(sum5) # upsample by a factor of 2
    upsampling1 = tf.keras.layers.Conv2D(128, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(upsampling1)
    concat1 = tf.keras.layers.concatenate([sum4, upsampling1])
    localization_1 = tf.keras.layers.Conv2D(128, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(concat1)
    localization_1 = tf.keras.layers.BatchNormalization()(localization_1) #BatchNormalization layers used to avoid overfitting
    localization_1 = tf.keras.layers.Conv2D(128, (1,1), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(localization_1)
    localization_1 = tf.keras.layers.BatchNormalization()(localization_1)
    upsampling2 = tf.keras.layers.UpSampling2D(size=(2,2))(localization_1)

    upsampling2 = tf.keras.layers.Conv2D(64, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(upsampling2)
    concat2 = tf.keras.layers.concatenate([sum3, upsampling2])
    localization_2 = tf.keras.layers.Conv2D(64, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(concat2)
    localization_2 = tf.keras.layers.BatchNormalization()(localization_2)
    localization_2 = tf.keras.layers.Conv2D(64, (1,1), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(localization_2)
    localization_2 = tf.keras.layers.BatchNormalization()(localization_2)
    segmentation1 = tf.keras.layers.Conv2D(1, (1,1))(localization_2)
    segmentation1 = tf.keras.layers.UpSampling2D(size=(2,2))(segmentation1)
    upsampling3 = tf.keras.layers.UpSampling2D(size=(2,2))(localization_2)

    upsampling3 = tf.keras.layers.Conv2D(32, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(upsampling3)
    concat3 = tf.keras.layers.concatenate([sum2, upsampling3])
    localization_3 = tf.keras.layers.Conv2D(32, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(concat3)
    localization_3 = tf.keras.layers.BatchNormalization()(localization_3)
    localization_3 = tf.keras.layers.Conv2D(32, (1,1), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(localization_3)
    localization_3 = tf.keras.layers.BatchNormalization()(localization_3)
    segmentation2 = tf.keras.layers.Conv2D(1, (1,1))(localization_3)
    segmentation2 = tf.keras.layers.Add()([segmentation1, segmentation2])
    segmentation2 = tf.keras.layers.UpSampling2D(size=(2,2))(segmentation2)
    upsampling4 = tf.keras.layers.UpSampling2D(size=(2,2))(localization_3)

    upsampling4 = tf.keras.layers.Conv2D(32, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(upsampling4)
    concat4 = tf.keras.layers.concatenate([sum1, upsampling4])
    conv6 = tf.keras.layers.Conv2D(32, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(concat4)
    segmentation3 = tf.keras.layers.Conv2D(1, (1,1))(conv6)
    segmentation3 = tf.keras.layers.Add()([segmentation3, segmentation2])
    output = tf.keras.layers.Conv2D(1, (1,1),  activation='sigmoid')(segmentation3) # sigmoid activation as we either have 0 or 1 for each pixel.

    model = tf.keras.Model(inputs=input, outputs=output)
    
    return model