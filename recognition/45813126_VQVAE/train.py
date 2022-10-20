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

        vae = VAE(self.num_embeddings, self.latent_dim)
        self.vqvae = vae.generate_model()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
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

num_embeddings = 128
latent_dim = 32
vae_trainer = VQVAETrainer(1, latent_dim, num_embeddings)
vae_trainer.compile(optimizer = 'adam')
vae_trainer.fit(data.train_data, epochs = 2, batch_size = 128)