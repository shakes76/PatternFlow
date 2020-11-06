import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2DTranspose\
    ,Conv2D, LeakyReLU, Dropout, Flatten
import matplotlib.pyplot as plt
from time import time
from os import path, mkdir
# Constants used in getting the result
NOISE_INPUT_DIM = 64
BATCH_SIZE = 90
LEARNING_RATE = 1e-4

class TrainLoader():
    def __init__(self, data_loader, argument_parser):
        self.dataset_loader = data_loader
        self.argument_parser = argument_parser
        self.binary_cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        
        # Setting up the generator and discriminator models
        self.generator = self._generator_model()
        self.discriminator = self._discriminator_model()
        self.generator_optmiser_func = self._generator_optimiser()
        self.discriminator_optimiser_func = self._discriminator_optimiser()
    
    def _generator_model(self):
        model = Sequential()
        model.add(Dense(4 * 4 * 256, use_bias=False, input_shape=(NOISE_INPUT_DIM,)))
        model.add(BatchNormalization())
        model.add(Reshape((4, 4, 256)))

        model.add(Conv2DTranspose(64, (5, 5), strides=(1, 1), padding="same", activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(8, (5, 5), strides=(2, 2), padding="same", activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(16, (32, 32), strides=(8, 8), padding="same", activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(1, (32, 32), strides=(4, 4), padding="same", activation="tanh"))

        return model

    def generator_loss_func(self, output):
        return self.binary_cross_entropy(tf.ones_like(output), output)

    def _generator_optimiser(self):
        return optimizers.Adam(LEARNING_RATE)

    def _discriminator_model(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(5, 5), padding="same", input_shape=[256, 256, 1]))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(5, 5), padding="same"))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))


        model.add(Conv2D(128, (5, 5), strides=(5, 5), padding="same"))
        model.add(LeakyReLU())
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1))

        return model

    def discriminator_loss_func(self, ro, fo):
        real_loss = self.binary_cross_entropy(tf.ones_like(ro), ro)
        fake_loss = self.binary_cross_entropy(tf.zeros_like(fo), fo)
        return real_loss + fake_loss

    def _discriminator_optimiser(self):
        return optimizers.Adam(LEARNING_RATE)

    @tf.function
    def _training_step(self, images):
        input_noise_set = tf.random.normal([BATCH_SIZE, NOISE_INPUT_DIM])
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_image = self.generator(input_noise_set, training=True)
            output_real = self.discriminator(images, training=True)
            output_fake = self.discriminator(generated_image, training=True)
            
            generator_current_loss = self.generator_loss_func(output_fake)
            discriminator_current_loss = self.discriminator_loss_func(output_real, output_fake)
        generator_gradient = generator_tape.gradient(generator_current_loss, self.generator.trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, self.discriminator.trainable_variables)
        # Apply gradient optmised to both generator and discriminator
        self.generator_optmiser_func \
                .apply_gradients(zip(generator_gradient, self.generator.trainable_variables))
        self.discriminator_optimiser_func \
                .apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))
    def train(self):
        dataset = self.dataset_loader.get_dataset()
        training_epoch = self.argument_parser.get_training_epoch()
        total_dataset = self.argument_parser.get_training_size()

        # Get the minimum just in case the length is greater than available
        total_dataset = min(total_dataset, len(dataset))

        if not path.isdir(path.abspath("./output")):
            print("CREATED output directory")
            mkdir(path.abspath("./output"))

        if not path.isdir(path.abspath("./images")):
            print("CREATED image directory")
            mkdir(path.abspath("./images"))
        
        try:
            for current_epoch in range(training_epoch):
                print("STARTED EPOCH {}".format(current_epoch))
                for idx in range(total_dataset):
                    image_dataset = dataset[idx]
                    expanded_image_dataset = tf.expand_dims(image_dataset, axis=0)
                    expanded_image_dataset = tf.expand_dims(expanded_image_dataset, axis=-1)
                    self._training_step(expanded_image_dataset)
                if current_epoch % 5 == 0:
                    self.generator.save(path.abspath("./output/generator"))
                    self.discriminator.save(path.abspath("./output/discriminator"))
                    # Test samples
                    noise = tf.random.normal([1, NOISE_INPUT_DIM])
                    generated_image = self.generator(noise)[0]
                    # Reverse normalisation
                    generated_image = (generated_image + 1) / 2.0
                    # Save the image
                    plt.imshow(generated_image, cmap="gray")
                    plt.savefig(path.abspath("./images/{}.png".format(current_epoch)))
        except:
            print("ERROR: Something went wrong during training")
