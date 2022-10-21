import dataset
import modules
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Trainer(tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim, num_embeddings, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = modules.VQVAE(self.latent_dim, self.num_embeddings)

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


def plot_losses(history, time):
    plt.figure()
    plt.plot(history.history["loss"], label='Combined Loss')
    plt.plot(history.history["reconstruction_loss"], label='Reconstruction Loss')
    plt.plot(history.history["vqvae_loss"], label='VQ VAE Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"out/{time}/training_loss_curves.png")
    plt.close()


def plot_reconstructions(trained_vqvae_model, dataset, time):
    num_tests = 8
    test_images = dataset[np.random.choice(len(dataset), 8)]
    reconstructions = trained_vqvae_model.predict(test_images)

    i = 0
    plt.figure(figsize=(4, num_tests * 2), dpi=512)
    for test_image, reconstructed_image in zip(test_images, reconstructions):
        plt.subplot(num_tests, 2, 2 * i + 1, )
        plt.imshow(test_image.squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(num_tests, 2, 2 * i + 2)
        plt.imshow(reconstructed_image[:, :, 0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")

        i += 1

    plt.savefig(f"out/{time}/reconstruction.png")
    plt.close()


def main():
    (train_data, validate_data, test_data, data_variance) = dataset.oasis_dataset(5000)
    time = datetime.now().strftime('%H:%M:%S')

    num_embeddings = 128
    latent_dim = 16
    batch_size = 8

    # VQ VAE
    vqvae_trainer = Trainer(data_variance, latent_dim=latent_dim, num_embeddings=num_embeddings)
    vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
    vqvae_trainer.vqvae.save(f"out/{time}/vqvae_model")

    history = vqvae_trainer.fit(train_data, epochs=10, batch_size=batch_size)

    plot_reconstructions(vqvae_trainer.vqvae, test_data, time)
    plot_losses(history, time)

    # PIXELCNN
    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

    encoded_outputs = encoder.predict(test_data)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    pixel_cnn = modules.PixelCNN(latent_dim, num_embeddings, 2, 2)
    pixel_cnn.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
    pixel_cnn.fit(x=codebook_indices, y=codebook_indices, batch_size=batch_size,
                  epochs=10, validation_split=0.2)

    trained_pixelcnn_model = pixel_cnn
    trained_pixelcnn_model.save(f"out/{time}/pixelcnn_model")


if __name__ == "__main__":
    main()
