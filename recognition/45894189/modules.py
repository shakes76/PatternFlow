import tensorflow as tf
from tensorflow import keras
from keras import layers

def add_noise(n_filters, image_size):
    """
    Noise input layer [B]
    """
    input = layers.Input((image_size, image_size, n_filters))
    noise = layers.Input((image_size, image_size, 1))
    x = input

    scaled_noise = layers.Dense(n_filters)(noise)
    x = x + scaled_noise
    return tf.keras.Model([input, noise], x)

def AdaIN(n_filters, image_size, epsilon=1e-8):
    """
    Combined AdaIN and [A] layer. Affine transform of W is used to perform
    Instance Normalisation on X values 
    """
    input = layers.Input((image_size, image_size, n_filters))
    w = layers.Input(512) # = latent_dim
    x = input

    ys = layers.Dense(n_filters)(w)
    ys = layers.Reshape([1, 1, n_filters])(ys)

    yb = layers.Dense(n_filters)(w)
    yb = layers.Reshape([1, 1, n_filters])(yb)

    axes = [1, 2]   # axes image pixel values, without batch or filters axes
    mean = tf.math.reduce_mean(x, axes, keepdims=True)
    stddev = tf.math.reduce_std(x, axes, keepdims=True) + epsilon

    x = ys*((x-mean)/(stddev + epsilon))
    return tf.keras.Model([input, w], x)

def WNetwork(latent_dim=512):
    """
    Mapping Network z -> w mapping latent noise to style code with set of
    fully connected layers
    returns: Mapping Network
    """
    z = layers.Input(shape=[latent_dim])
    w = z
    for _ in range(8):
        w = layers.Dense(latent_dim)(z)
        w = layers.LeakyReLU(0.2)(z)
    return tf.keras.Model(z, w)

class Generator():
    def __init__(self):
        self.init_size = 4
        self.init_filters = 512

    def generator_block(self, n_filters, image_size):
        input_tensor = layers.Input(shape=(image_size, image_size, n_filters))
        noise = layers.Input(shape=(2*image_size, 2*image_size, 1))
        w = layers.Input(shape=512)

        x = input_tensor
        n_filters = n_filters // 2

        x = layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)

        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = add_noise(n_filters, image_size*2)([x, noise])
        x = AdaIN(n_filters, image_size*2)([x, w])
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = add_noise(n_filters, image_size*2)([x, noise])
        x = AdaIN(n_filters, image_size*2)([x, w])
        x = layers.LeakyReLU(alpha=0.2)(x)

        return tf.keras.Model([input_tensor, w, noise], x)

    def generator(self):
        current_size = self.init_size # 4
        n_filters = self.init_filters # 512
        input = layers.Input(shape=(current_size, current_size, n_filters))
        x = input
        i = 0

        noise_inputs, z_inputs = [], []
        curr_size = self.init_size
        while curr_size <= 256:
            noise_inputs.append(layers.Input(shape=[curr_size, curr_size, 1]))
            z_inputs.append(layers.Input(shape=[512]))
            curr_size *= 2

        z = layers.Input(shape=[512])
        w = z
        for _ in range(8):
            w = layers.Dense(n_filters)(w)
            w = layers.LeakyReLU(0.2)(w)
        mapping = tf.keras.Model(z, w)

        x = layers.Activation("linear")(x)    
        x = add_noise(n_filters, current_size)([x, noise_inputs[i]])
        x = AdaIN(n_filters, current_size)([x, mapping(z_inputs[i])])
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(512, kernel_size=3, padding="same")(x)
        x = add_noise(n_filters, current_size)([x, noise_inputs[i]])
        x = AdaIN(n_filters, current_size)([x, mapping(z_inputs[i])])
        x = layers.LeakyReLU(alpha=0.2)(x)

        while current_size < 256:
            i += 1
            x = self.generator_block(n_filters, current_size)([x, mapping(z_inputs[i]), noise_inputs[i]])
            current_size = 2 * current_size
            n_filters = n_filters // 2

        x = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")(x)
        return tf.keras.Model([input, z_inputs, noise_inputs], x)

class Discriminator():

    def __init__(self):
        self.init_size = 256
        self.init_filters = 8

    def discriminator_block(self, n_filters, image_size):
        """
        Main block for discriminator containing two convolutional layers with LeakyReLU activation. the second layer doubles the number
        of filters. The block is ended with a downsampling of the image that halves it's size
        """

        if image_size == self.init_size:
            input_tensor = layers.Input(shape=(image_size, image_size, 1))
        else:
            input_tensor = layers.Input(shape=(image_size, image_size, n_filters // 2))

        x = input_tensor
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.AveragePooling2D((2,2))(x)   # Downsizing 
        x = layers.LeakyReLU(0.2)(x)
        return tf.keras.Model(input_tensor, x)

    def discriminator(self):
        """
        Discriminator model, takes in 256x256x1 images and calssifies them as real or fake. Built with initial input and convolution layer,
        then repeated discriminator blocks until the image is downsampled to 4x4, and finished with 2 convolutions layers, then a flatten and
        Dense classification layer
        """
        current_size = self.init_size
        n_filters = self.init_filters
        input_tensor = layers.Input(shape=[current_size, current_size, 1])
        x = input_tensor

        while current_size > 4:
            x = self.discriminator_block(n_filters, current_size)(x)
            current_size = current_size // 2
            n_filters = 2 * n_filters

        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(input_tensor, x)