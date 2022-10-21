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
    num_epochs = len(history.history["loss"])
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

# def progress_images(dataset, model, time, epoch):
    # num_examples_to_generate = 8
    # test_images = dataset[np.random.choice(len(dataset), num_examples_to_generate)]
    # reconstructions = model.predict(test_images)
    #
    # plt.figure()
    # for i in range(reconstructions.shape[0]):
    #     plt.subplot(4, 4, 2*i + 1)
    #     plt.imshow(test_images[i, :, :, 0], cmap='gray')
    #     plt.imshow(reconstructions[i, :, :, 0], cmap='gray')
    #     plt.axis('off')
    #
    # plt.savefig(f"out/{time}image_at_epoch_{epoch.png")
    # plt.close()


def plot_reconstructions(trained_vqvae_model, dataset, time):
    num_tests = 8
    test_images = dataset[np.random.choice(len(dataset), 8)]
    reconstructions = trained_vqvae_model.predict(test_images)

    i = 0
    plt.figure(figsize=(4, num_tests*2), dpi=512)
    for test_image, reconstructed_image in zip(test_images, reconstructions):
        plt.subplot(num_tests, 2, 2*i + 1,)
        plt.imshow(test_image.squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(num_tests, 2, 2*i + 2)
        plt.imshow(reconstructed_image[:, :, 0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")

        i += 1

    plt.savefig(f"out/{time}/reconstruction.png")
    plt.close()


def main():
    (train_data, validate_data, test_data, data_variance) = dataset.oasis_dataset()
    time = datetime.now().strftime('%H:%M:%S')

    vqvae_trainer = Trainer(data_variance, latent_dim=16, num_embeddings=128)
    vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
    vqvae_trainer.vqvae.save(f"out/{time}/vqvae_model")

    history = vqvae_trainer.fit(train_data, epochs=10, batch_size=8)

    plot_reconstructions(vqvae_trainer.vqvae, test_data, time)
    plot_losses(history, time)


if __name__ == "__main__":
    main()
