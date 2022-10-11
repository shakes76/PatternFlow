"""
train.py

Alex Nicholson (45316207)
11/10/2022

Contains the source code for training, validating, testing and saving your model. The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make sure to plot the losses and metrics during training.

"""


import dataset
import modules
import utils
from tensorflow import keras
import tensorflow as tf
import numpy as np # TODO: Remove this later once we swap to the OASIS data


class VQVAETrainer(keras.models.Model):
    """
    A Custom Training Loop for the VQ-VAE Model

        Attributes:
            train_variance (ndarray): The input data for the model (input data in the form of variances?) ???
            latent_dim (int): The number of latent dimensions the images are compressed down to (default=32)
            num_embeddings (int): The number of codebook vectors in the embedding space (default=128)
            vqvae (Keras Model): The custom VQ-VAE model
            total_loss_tracker (Keras Metric): A tracker for the total loss performance of the model during training???
            reconstruction_loss_tracker (Keras Metric): A tracker for the reconstruction loss performance of the model during training???
            vq_loss_tracker (Keras Metric): A tracker for the VQ loss performance of the model during training???

        Methods:
            metrics(): Returns a list of metrics for the total_loss, reconstruction_loss, and vq_loss of the model
            train_step(x): Trains the model for a single step using the given training sample/samples x???
    """

    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = modules.get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        """
        Gets a list of metrics for current total_loss, reconstruction_loss, and vq_loss of the model
        
            Returns:
                A list of metrics for total_loss, reconstruction_loss, and vq_loss
        """

        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        """
        Trains the model for a single step using the given training sample/samples x???

            Parameters:
                x (Tensorflow Tensor???): The input training sample/samples (how big is a training step? how many samples?) ???

            Returns:
                A dictionary of the model's training metrics with keys: "loss", "reconstruction_loss", and "vqvae_loss"
        """

        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    # (train_data, test_data, validate_data) = dataset.load_dataset()


    # ------------------------------------ NEW ----------------------------------- #
    # Load and preprocess the MNIST dataset
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5

    data_variance = np.var(x_train / 255.0)
    # ------------------------------------ NEW ----------------------------------- #


    # ---------------------------------------------------------------------------- #
    #                                  BUILD MODEL                                 #
    # ---------------------------------------------------------------------------- #
    # Create the model (wrapped in the training class to handle performance metrics logging)
    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())



    # ---------------------------------------------------------------------------- #
    #                                 RUN TRAINING                                 #
    # ---------------------------------------------------------------------------- #
    print("Training model...")
    # Run training, plotting losses and metrics throughout
    vqvae_trainer.fit(x_train_scaled, epochs=1, batch_size=128)


    # ---------------------------------------------------------------------------- #
    #                                SAVE THE MODEL                                #
    # ---------------------------------------------------------------------------- #
    # Get the trained model
    trained_vqvae_model = vqvae_trainer.vqvae

    # Save the model to file as a tensorflow SavedModel
    trained_vqvae_model.save('./vqvae_saved_model')

    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    # Visualise the final results and calculate the structural similarity index (SSIM)

    # ------------------------------------ NEW ----------------------------------- #
    idx = np.random.choice(len(x_test_scaled), 10)
    test_images = x_test_scaled[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        utils.show_subplot(test_image, reconstructed_image)
    # ------------------------------------ NEW ----------------------------------- #


