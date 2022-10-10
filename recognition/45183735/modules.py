import tensorflow as tf


# Mapping Network
class G_Mapping:

    def __init__(self, latent_size):
        self.latent_size = latent_size
        # Mapping network for generator
        self.model = self.mapping_nw()

    # build the mapping network
    def mapping_nw(self):
        fc_input = tf.keras.layers.Input(shape=(self.latent_size,))
        fc1 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc_input)
        fc2 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc1)
        fc3 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc2)
        fc4 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc3)
        fc5 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc4)
        fc6 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc5)
        fc7 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc6)
        return tf.keras.Model(inputs=[fc_input], outputs=[fc7])

# Synthesis network
class G_Synthesis:
    def __init__(self, latent_size, g_mapping, input_size):
        # input image size
        self.input_size = input_size
        self.latent_size = latent_size
        # Latent inputs
        self.z = self.get_latent_inputs()
        # Mapping network
        self.g_mapping = g_mapping
        # non_liner mapping network to map z -> w
        self.w = self.get_w()
        # Noises input
        self.noises_input = self.get_noises_input()
        # Noises
        self.noises = self.get_noises()
        # Synthesis network
        self.nw = self.get_synthesis_nw()

    def get_latent_inputs(self):
        z = []
        for i in range(7):
            z.append(tf.keras.layers.Input(shape=(self.latent_size,)))
        return z

    def get_w(self):
        w = []
        for i in range(7):
            w.append(self.g_mapping.model(self.z[i]))
        return w

    def get_noises_input(self):
        noises_input = []
        for i in range(7):
            noises_input.append(tf.keras.layers.Input(shape=(4 * 2 ** i, 4 * 2 ** i, 1)))
        return noises_input

    def get_noises(self):
        noises = []
        for i in range(7):
            noises.append(tf.keras.layers.Dense(32, activation="relu")(self.noises_input[i]))
        return noises

    # Adaptive instance normalisation
    def get_AdaIN(self, x, ys, yb):
        x_mean, x_std = tf.keras.backend.mean(x), tf.keras.backend.std(x)
        ys = tf.reshape(ys, (-1, 1, 1, tf.shape(ys)[-1]))
        yb = tf.reshape(yb, (-1, 1, 1, tf.shape(yb)[-1]))
        return tf.add(tf.multiply(ys, tf.divide(x - x_mean, x_std + 1e-7)), yb)

    def get_synthesis_nw(self):
        layer = tf.keras.layers.Dense(4 * 4 * 32, activation="relu")(self.z[0])
        layer = tf.keras.layers.Reshape((4, 4, 32))(layer)
        noise_b = tf.keras.layers.Dense(32)(self.noises[0])
        # add noise
        layer = tf.keras.layers.Add()([layer, noise_b])
        # add the style in AdaIN
        layer = self.get_AdaIN(layer, tf.keras.layers.Dense(32)(self.w[0]), tf.keras.layers.Dense(32)(self.w[0]))
        layer = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(layer)
        # add noise
        layer = tf.keras.layers.Add()([layer, noise_b])
        # add the style in AdaIN
        layer = self.get_AdaIN(layer, tf.keras.layers.Dense(32)(self.w[0]), tf.keras.layers.Dense(32)(self.w[0]))

        # for 8x8 to 256x256
        for i in range(6):
            layer = tf.keras.layers.UpSampling2D()(layer)
            layer = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(layer)
            noise_b = tf.keras.layers.Dense(32)(self.noises[i + 1])
            layer = tf.keras.layers.Add()([layer, noise_b])
            layer = self.get_AdaIN(layer, tf.keras.layers.Dense(32)(self.w[i + 1]),
                                   tf.keras.layers.Dense(32)(self.w[i + 1]))
            layer = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(layer)
            layer = tf.keras.layers.Add()([layer, noise_b])
            layer = self.get_AdaIN(layer, tf.keras.layers.Dense(32)(self.w[i + 1]),
                                   tf.keras.layers.Dense(32)(self.w[i + 1]))

        layer = tf.keras.layers.Dense(1)(layer)
        layer = tf.keras.layers.Activation("sigmoid")(layer)
        return layer


# generator model
class G_style:

    def __init__(self, latent_size, input_size, g_synthesis):
        self.input_size = input_size
        self.latent_size = latent_size
        self.g_synthesis = g_synthesis
        self.model = self.generation_model()

    def generation_model(self):
        model = tf.keras.Model(inputs=self.g_synthesis.z + self.g_synthesis.noises_input, outputs=[self.g_synthesis.nw])
        model.summary()
        return model


# as the styleGan does not modify discriminator in any way, so we using the discriminator structure of PGGan
class Discriminator:

    def __init__(self, input_size):
        self.input_size = input_size
        self.d_model = self.generate_discriminator_model()

    def generate_discriminator_model(self):
        D_model = tf.keras.models.Sequential()

        # 256x256 -> 128x128
        D_model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                                           input_shape=[self.input_size, self.input_size, 1]))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Dropout(0.2))

        # 128x128 -> 64x64
        D_model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Dropout(0.3))

        # 64x64 -> 32x32
        D_model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Dropout(0.4))

        # 32x32 -> 16x16
        D_model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Dropout(0.4))

        # 16x16 -> 8x8
        D_model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Dropout(0.4))

        # 8x8 -> 4x4
        D_model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        D_model.add(tf.keras.layers.LeakyReLU())
        D_model.add(tf.keras.layers.Dropout(0.4))

        D_model.add(tf.keras.layers.Flatten())
        D_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        return D_model


# if __name__ == "__main__":
#     latent_size = 512
#     input_size = 256
#     # g_mapping = G_Mapping(latent_size)
#     # g_s = G_Synthesis(latent_size, g_mapping, input_size)
#     # g_style = G_style(latent_size, input_size, g_s)
#     discriminator = Discriminator(input_size)
#     discriminator.D_model.summary()

