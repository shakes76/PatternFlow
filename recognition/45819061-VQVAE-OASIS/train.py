from matplotlib import pyplot as plt
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


x_train, x_test, x_validate, mean, data_variance = get_data()

data_variance = np.var(x_train / 255.0)

vqvae_trainer = VQVAETrainer(data_variance, 16, 64)
vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
vqvae_trainer.fit(x_train, epochs=2, batch_size=BATCH_SIZE, use_multiprocessing=True)

vqvae_trainer.model.save('mymodel')

def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()


trained_vqvae_model = vqvae_trainer.model
idx = np.random.choice(len(x_test), 10)
test_images = x_test[idx]
reconstructions_test = trained_vqvae_model.predict(test_images)

for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    show_subplot(test_image, reconstructed_image)

encoder = vqvae_trainer.model.get_layer("encoder")
quantizer = vqvae_trainer.model.get_layer("vector_quantizer")

encoded_outputs = encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

for i in range(len(test_images)):
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i].squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices[i])
    plt.title("Code")
    plt.axis("off")
    plt.show()