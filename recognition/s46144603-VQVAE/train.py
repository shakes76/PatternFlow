from modules import *

# Training loop for VQVAE with backpropagation and loss tracking 
# Source: https://keras.io/examples/generative/vq_vae/
class VQVAETrainer(tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim=64, num_embeddings=256, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.vqvae = vq_vae(self.latent_dim, self.num_embeddings)

        self.track_total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.track_reconstructionloss = tf.keras.metrics.Mean(
            name="recon_loss"
        )
        self.track_vqloss= tf.keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.track_total_loss,
            self.track_reconstructionloss,
            self.track_vqloss,
        ]

    def train_step(self, input):
        with tf.GradientTape() as tape:
            # VQ-VAE outputs
            reconstructions = self.vqvae(input)

            # Calculate losses
            reconstruction_loss = (
                tf.reduce_mean((input - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation
        gradients = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.vqvae.trainable_variables))

        # Track losses
        self.track_total_loss.update_state(total_loss)
        self.track_reconstructionloss.update_state(reconstruction_loss)
        self.track_vqloss.update_state(sum(self.vqvae.losses))

        # Log results
        return {
            "loss": self.track_total_loss.result(),
            "recon_loss": self.track_reconstructionloss.result(),
            "vqvae_loss": self.track_vqloss.result(),
        }

# Model 
train_vqvae = VQVAETrainer(data_variance, latent_dim=64, num_embeddings=256)
train_vqvae.compile(optimizer=tf.keras.optimizers.Adam(), metrics="loss")
train_vqvae.fit(X, epochs=200, batch_size=2)

# Plot model loss
plt.plot(train_vqvae.history.history['loss'])
plt.plot(train_vqvae.history.history['recon_loss'])
plt.plot(train_vqvae.history.history['vqvae_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['total loss', 'recon loss', 'vae loss'], loc='upper left')
plt.show