import dataset
import modules
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt


class Trainer(tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim, num_embeddings, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = modules.VQVAE(self.latent_dim, self.num_embeddings)

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
            reconstructions = self.vqvae(x)

            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }


def main():
    (train_data, validate_data, test_data, data_variance) = dataset.oasis_dataset(images=100)

    vqvae_trainer = Trainer(data_variance, latent_dim=16, num_embeddings=128)
    vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
    vqvae_trainer.vqvae.save(f"out/vqvae_model_{datetime.now().strftime('%H:%M:%S')}")

    history = vqvae_trainer.fit(train_data, epochs=3, batch_size=4)


if __name__ == "__main__":
    main()
