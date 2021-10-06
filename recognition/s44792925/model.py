import tensorflow as tf

def context_module(input, conv):
    """Each context module is in fact a pre-activation residual block [13] with two
    3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between."""

    pass

def model(height, width, channel):
    input = tf.keras.layers.Input((height, width, channel))
    #encoding
    conv1 = tf.keras.layers.Conv2D(16, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(input)
    context_module1 = context_module(conv1, 16)
    sum1 = tf.keras.layers.Add([conv1, context_module1])

    conv2 = tf.keras.layers.Conv2D(32, (3,3), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(sum1)
    context_module2 = context_module(conv2, 32)
    sum2 = tf.keras.layers.Add([conv2, context_module2])
    
    conv3 = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(sum2)
    context_module3 = context_module(conv3, 64)
    sum3 = tf.keras.layers.Add([conv3, context_module3])

    conv4 = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(sum3)
    context_module4 = context_module(conv4, 128)
    sum4 = tf.keras.layers.Add([conv4, context_module4])

    conv5 = tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(sum4)
    context_module5 = context_module(conv5, 256)
    sum5 = tf.keras.layers.Add([conv4, context_module5])
    
    #decoding
    upsampling1 = tf.keras.layers.UpSampling2D(size=(2,2))(sum5)
    
    #possibly need a conv2d here
    concat1 = tf.keras.layers.concatenate([sum4, upsampling1])
    localization_1 = tf.keras.layers.Conv2D(128, (3,3), activation = tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(concat1)
    localization_1 = tf.keras.layers.Conv2D(128, (1,1), activation = tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(localization_1)
    upsampling2 = tf.keras.layers.UpSampling2D(size=(2,2))(localization_1)

    concat2 = tf.keras.layers.concatenate([sum3, upsampling2])
    localization_2 = tf.keras.layers.Conv2D(64, (3,3), activation = tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(concat2)
    localization_2 = tf.keras.layers.Conv2D(64, (1,1), activation = tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(localization_2)
    upsampling3 = tf.keras.layers.UpSampling2D(size=(2,2))(localization_2)

    concat3 = tf.keras.layers.concatenate([sum2, upsampling3])
    localization_3 = tf.keras.layers.Conv2D(32, (3,3), activation = tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(concat3)
    localization_3 = tf.keras.layers.Conv2D(32, (1,1), activation = tf.keras.layers.LeakyReLU(alpha=10**-2), padding ='same')(localization_3)
    upsampling4 = tf.keras.layers.Upsampling2D(size=(2,2))(localization_3) # upscale by factor of 2

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 5*(10**-4)), loss=["binary_crossentropy"], metrics=["accuracy"])
    return model