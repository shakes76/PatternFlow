"""
train.py

Alex Nicholson (45316207)
11/10/2022

Contains the source code for training, validating, testing and saving your model. The model is imported from “modules.py” and the data loader is imported from “dataset.py”. Losses and metrics are plotted throughout training.

"""


import dataset
import modules
import utils
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio.v2 as imageio


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
            vqvae_loss_tracker (Keras Metric): A tracker for the VQ loss performance of the model during training???

        Methods:
            metrics(): Returns a list of metrics for the total_loss, reconstruction_loss, and vqvae_loss of the model
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
        self.vqvae_loss_tracker = keras.metrics.Mean(name="vqvae_loss")
        self.ssim_history = []

    @property
    def metrics(self):
        """
        Gets a list of metrics for current total_loss, reconstruction_loss, and vqvae_loss of the model
        
            Returns:
                A list of metrics for total_loss, reconstruction_loss, and vqvae_loss
        """

        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vqvae_loss_tracker,
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
        self.vqvae_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vqvae_loss_tracker.result(),
        }


class ProgressImagesCallback(keras.callbacks.Callback):
    """
    A custom callback for saving training progeress images
    """

    def __init__(self, train_data):
        self.train_data = train_data

    def save_progress_image(self, epoch):
        """
        Saves progress images as we go throughout training

            Parameters:
                epoch (int): The current training epoch
        """

        num_examples_to_generate = 16
        idx = np.random.choice(len(self.train_data), num_examples_to_generate)
        test_images = self.train_data[idx]
        reconstructions_test = self.model.vqvae.predict(test_images)

        fig = plt.figure(figsize=(16, 16))
        for i in range(reconstructions_test.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(reconstructions_test[i, :, :, 0], cmap='gray')
            plt.axis('off')

        plt.savefig('out/image_at_epoch_{:04d}.png'.format(epoch+1))
        plt.close()

    def create_gif(self):
        """
        Show an animated gif of the progress throughout training
        """

        anim_file = 'vqvae_training_progression.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('out/image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                image = imageio.imread(filename)
                writer.append_data(image)

    def on_epoch_end(self, epoch, logs=None):
        self.save_progress_image(epoch)
        
        similarity = utils.get_model_ssim(self.model.vqvae, test_data)
        self.model.ssim_history.append(similarity)
        print(f"ssim: {similarity}")

    def on_train_end(self, logs=None):
        self.create_gif()


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                HYPERPARAMETERS                               #
    # ---------------------------------------------------------------------------- #
    NUM_TRAINING_EXAMPLES = None

    TRAINING_EPOCHS = 10
    BATCH_SIZE = 128

    NUM_LATENT_DIMS = 16
    NUM_EMBEDDINGS = 128

    EXAMPLES_TO_SHOW = 10


    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    (train_data, validate_data, test_data, data_variance) = dataset.load_dataset(max_images=NUM_TRAINING_EXAMPLES, verbose=True)


    # ---------------------------------------------------------------------------- #
    #                                  BUILD MODEL                                 #
    # ---------------------------------------------------------------------------- #
    # Create the model (wrapped in the training class to handle performance metrics logging)
    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=NUM_LATENT_DIMS, num_embeddings=NUM_EMBEDDINGS)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())


    # ---------------------------------------------------------------------------- #
    #                                 RUN TRAINING                                 #
    # ---------------------------------------------------------------------------- #
    print("Training model...")
    # Run training, plotting losses and metrics throughout
    history = vqvae_trainer.fit(train_data, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, callbacks=[ProgressImagesCallback(train_data)])


    # ---------------------------------------------------------------------------- #
    #                                SAVE THE MODEL                                #
    # ---------------------------------------------------------------------------- #
    # Get the trained model
    trained_vqvae_model = vqvae_trainer.vqvae

    # Save the model to file as a tensorflow SavedModel
    trained_vqvae_model.save("vqvae_saved_model")


    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    # Visualise the model training curves
    utils.plot_training_metrics(history)
    utils.plot_ssim_history(vqvae_trainer.ssim_history)
    
    # Visualise output generations from the finished model
    # utils.show_reconstruction_examples(trained_vqvae_model, validate_data, EXAMPLES_TO_SHOW)