"""
    File name : trainer.py
    Author : Arun Harish Balasubramonian
    Student Number : 44340326
    Description : Trainer module that takes the image stored on memory to 
                  train the model. It saves periodically for every 5th
                  iteration along with the saved image.
"""

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

"""
    The train loader is responsible for training the model by getting the command line argument
    options relevant to epoch and number of dataset to be used. It saves the model and its output image
    on every fifth epoch.
    Reference : The code is similar to https://www.tensorflow.org/tutorials/generative/dcgan, but adapted
    to the OASIS dataset.
"""
class TrainLoader():
    def __init__(self, data_loader, argument_parser):
        # Sets both the argument parser, to retrieve command line arguments, and
        # the data loader to get the image loaded. 
        self.dataset_loader = data_loader
        self.argument_parser = argument_parser
        # From logits set to True for sigmoid activation function to be used
        # avoiding precision or numerical overflow errors
        self.binary_cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        
        # Setting up the generator and discriminator models and its optimisers
        self.generator = self._generator_model()
        self.discriminator = self._discriminator_model()
        self.generator_optmiser_func = self._generator_optimiser()
        self.discriminator_optimiser_func = self._discriminator_optimiser()
    
    # An internal function that generates the generator model
    def _generator_model(self):
        model = Sequential()
        # Since the image is 256, 256
        model.add(Dense(4 * 4 * 256, use_bias=False, input_shape=(NOISE_INPUT_DIM,)))
        model.add(BatchNormalization())
        model.add(Reshape((4, 4, 256)))
        # Using relu for generator model
        model.add(Conv2DTranspose(64, (5, 5), strides=(1, 1), padding="same", activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(8, (5, 5), strides=(2, 2), padding="same", activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(16, (32, 32), strides=(8, 8), padding="same", activation="relu"))
        model.add(BatchNormalization())
        # Finally using tanh for activation function
        model.add(Conv2DTranspose(1, (32, 32), strides=(4, 4), padding="same", activation="tanh"))

        return model

    # Binary Cross entropy loss function for generator
    def generator_loss_func(self, output):
        return self.binary_cross_entropy(tf.ones_like(output), output)
    
    # Using Adam optmisier
    def _generator_optimiser(self):
        return optimizers.Adam(LEARNING_RATE)

    # To generate discriminator model
    def _discriminator_model(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(5, 5), padding="same", input_shape=[256, 256, 1]))
        # leaky relu
        model.add(LeakyReLU())
        # 30% Dropout
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(5, 5), padding="same"))
        # leaky relu
        model.add(LeakyReLU())
        # 30% Dropout
        model.add(Dropout(0.3))

        # Upsampling convolutional layer
        model.add(Conv2D(128, (5, 5), strides=(5, 5), padding="same"))
        # leaky relu
        model.add(LeakyReLU())
        # 25% Dropout
        model.add(Dropout(0.25))
        
        # Flatten the layer to dense
        model.add(Flatten())

        # Only has one classifier, binary class label (real or fake)
        model.add(Dense(1))

        return model


    # Discriminator loss function that takes the real and fake output by the discriminator
    # follows the function log D(x) + log(1 - D(G(z)))
    def discriminator_loss_func(self, ro, fo):
        real_loss = self.binary_cross_entropy(tf.ones_like(ro), ro)
        fake_loss = self.binary_cross_entropy(tf.zeros_like(fo), fo)
        return real_loss + fake_loss


    # Adam optmisier for discriminator
    def _discriminator_optimiser(self):
        return optimizers.Adam(LEARNING_RATE)

    # Function to update the gradient and the loss function of both
    # generator and discriminiator
    @tf.function
    def _training_step(self, images):
        # Random from normal distribution
        input_noise_set = tf.random.normal([BATCH_SIZE, NOISE_INPUT_DIM])
        # Setting gradient tapes to update the gradient and to apply loss function
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # Generate an image from the generator
            generated_image = self.generator(input_noise_set, training=True)
            # Get the classifier of the real image
            output_real = self.discriminator(images, training=True)
            # Get the classifier of the fake image
            output_fake = self.discriminator(generated_image, training=True)
            # Generator loss
            generator_current_loss = self.generator_loss_func(output_fake)
            # Discriminator loss
            discriminator_current_loss = self.discriminator_loss_func(output_real, output_fake)
        # Update the generator gradient
        generator_gradient = generator_tape.gradient(generator_current_loss, self.generator.trainable_variables)
        # Update the discriminator gradient
        discriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, self.discriminator.trainable_variables)
        # Apply gradient optmised to both generator and discriminator
        self.generator_optmiser_func \
                .apply_gradients(zip(generator_gradient, self.generator.trainable_variables))
        self.discriminator_optimiser_func \
                .apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))
    
    # Load the dataset and train the models on them
    def train(self):
        # Fetching dataset from the dataset loader
        dataset = self.dataset_loader.get_dataset()
        # Get the total training epoch from the user
        training_epoch = self.argument_parser.get_training_epoch()
        # Get the total image dataset to be trained on from the user
        total_dataset = self.argument_parser.get_training_size()
        # Get the minimum just in case the length is greater than available
        total_dataset = min(total_dataset, len(dataset))
        # If not output directory is found
        if not path.isdir(path.abspath("./output")):
            print("CREATED output directory")
            mkdir(path.abspath("./output"))
        # If not image directory is found
        if not path.isdir(path.abspath("./images")):
            print("CREATED image directory")
            mkdir(path.abspath("./images"))
        
        try:
            # The last epoch before ending the training
            # Is set during the end of every epoch
            # Used to save the image after the end of training image
            final_epoch = None
            # For every epoch 
            for current_epoch in range(training_epoch):
                print("STARTED EPOCH {}".format(current_epoch))
                # For every images selected
                for idx in range(total_dataset):
                    image_dataset = dataset[idx]
                    expanded_image_dataset = tf.expand_dims(image_dataset, axis=0)
                    expanded_image_dataset = tf.expand_dims(expanded_image_dataset, axis=-1)
                    self._training_step(expanded_image_dataset)
                # For every fifth epoch
                if current_epoch % 5 == 0:
                    self.generator.save(path.abspath("./output/generator"))
                    self.discriminator.save(path.abspath("./output/discriminator"))
                    # Test samples
                    noise = tf.random.normal([1, NOISE_INPUT_DIM])
                    generated_image = self.generator(noise)[0]
                    # Reverse normalisation
                    generated_image = (generated_image + 1) / 2.0
                    # Save the image
                    image_title = "{}.png".format(current_epoch)
                    # Saving epoch
                    print("Saving image:{}".format(image_title))
                    # Setting title
                    plt.title("EPOCH {}".format(current_epoch))
                    plt.imshow(generated_image, cmap="gray")
                    # Saving the generated image of this epoch
                    plt.savefig(path.abspath("./images/{}".format(image_title)))
                # Setting this epoch as the final epoch
                final_epoch = current_epoch
            # Comes here in the very end after the end of epoch iteration
            # Useful especially when the epoch is divisible by 5
            if final_epoch is not None:
                # Save it finally
                self.generator.save(path.abspath("./output/generator"))
                self.discriminator.save(path.abspath("./output/discriminator"))
                # Test samples
                noise = tf.random.normal([1, NOISE_INPUT_DIM])
                generated_image = self.generator(noise)[0]
                # Reverse normalisation
                generated_image = (generated_image + 1) / 2.0
                # Save the final image
                image_title = "{}.png".format(final_epoch + 1)
                # Saving epoch
                print("Saving image:{}".format(image_title))
                # Save the image
                plt.imshow(generated_image, cmap="gray")
                plt.savefig(path.abspath("./images/{}".format(image_title)))
        # Indicate something went wrong in training for debugging
        except:
            print("ERROR: Something went wrong during training")
