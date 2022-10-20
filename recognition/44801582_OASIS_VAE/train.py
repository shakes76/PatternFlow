import dataset
import modules
from datetime import datetime
import tensorflow as tf


class Trainer(tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim, num_embeddings, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = modules.VQVAE(self.latent_dim, self.num_embeddings)

        self.trackers = {
            "total_loss": tf.keras.metrics.Mean(name="total_loss"),
            "reconstruction_loss": tf.keras.metrics.Mean(name="reconstruction_loss"),
            "vq_vae_loss": tf.keras.metrics.Mean(name="vqvae_loss")
        }

    def metrics(self):
        self.trackers.values()

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructions = self.vqvae(x)

            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        self.trackers["total_loss"].update_state(total_loss)
        self.trackers["reconstruction_loss"].update_state(reconstruction_loss)
        self.trackers["vq_vae_loss"].update_state(sum(self.vqvae.losses))

        return self.trackers


def main():
    (train_data, validate_data, test_data, data_variance) = dataset.oasis_dataset(images=100)

    vqvae_trainer = Trainer(data_variance, latent_dim=16, num_embeddings=128)
    vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
    vqvae_trainer.vqvae.save(f"out/vqvae_model_{datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
