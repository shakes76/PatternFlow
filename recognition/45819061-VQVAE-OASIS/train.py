import tensorflow as tf
import numpy as np
from dataset import BATCH_SIZE, get_data
from modules import VQVAE


class VQVAETrainer (tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128):
        super(VQVAETrainer, self).__init__()
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.model = VQVAE(self.latent_dim, self.num_embeddings, (256, 256, 1))
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss_tracker = tf.keras.metrics.Mean(name='vq_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructions = self.model(x)

            reconstruction_loss = (tf.reduce_mean((x - reconstructions)**2)/self.train_variance)
            total_loss = reconstruction_loss + sum(self.model.losses)

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.model.losses))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result()
        }


x_train, x_test, x_validate, mean, variance = get_data()

vqvae_trainer = VQVAETrainer(variance, 32, 128)
vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
vqvae_trainer.fit(x_train, epochs=5, batch_size=BATCH_SIZE, use_multiprocessing=True, validation_data=x_validate)
vqvae_trainer.evaluate(x_test)