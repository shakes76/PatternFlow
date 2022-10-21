"""
File: train.py
Author: Georgia Spanevello
Student ID: 45813126
Description: Contains the classes required for training, validating, testing and saving the model.
"""

from dataset import Dataset
from modules import VAE
import tensorflow as tf
import matplotlib.pyplot as plt

# https://keras.io/examples/generative/vq_vae/#wrapping-up-the-training-loop-inside-vqvaetrainer
class VQVAETrainer(tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim = 32, num_embeddings = 128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = VAE(self.num_embeddings, self.latent_dim).model

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
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

# Load data
data = Dataset()

# Train model
num_embeddings = 128
latent_dim = 32
vae_trainer = VQVAETrainer(1, latent_dim, num_embeddings)
vae_trainer.compile(optimizer = 'adam')
history = vae_trainer.fit(data.train_data, epochs = 5, batch_size = 128)
trained_model = vae_trainer.vqvae

# Plot of loss metrics
plt.plot(history.history['loss'])
plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['vqvae_loss'])
plt.title('Model losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['total_loss', 'rec_loss', 'vqvae_loss'], loc='upper left')
plt.show()