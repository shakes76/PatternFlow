import tensorflow as tf

def context_module(input, conv):
    """Each context module is in fact a pre-activation residual block [13] with two
    3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between."""

    pass

def model(height, width, channel):
    input = tf.keras.layers.Input((height, width, channel))
    conv1 = tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(input)
    context_module1 = context_module(conv1, 16)
    sum1 = tf.keras.layers.Sum([conv1, context_module1])
    conv2 = tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), activation=tf.keras.layers.LeakyReLU(alpha=10**-2), padding='same')(sum1)
    #model = tf.keras.models.Sequential()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 5*(10**-4)), loss=["binary_crossentropy"], metrics=["accuracy"])
    return model