import os
import dataset
import modules
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


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
            "vqvae_loss": self.vq_loss_tracker.result()
        }


def plot_losses(history, time, name):
    plt.figure()
    for hist in history.history.keys():
        if hist == "accuracy" or hist == "val_accuracy":
            continue
        plt.plot(history.history[hist], label=hist)

    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"out/{time}/training_loss_curves_{name}.png")
    plt.close()


def plot_reconstructions(trained_vqvae_model, dataset, time):
    num_tests = 8
    test_images = dataset[np.random.choice(len(dataset), 8)]
    reconstructions = trained_vqvae_model.predict(test_images)

    i = 0
    plt.figure(figsize=(4, num_tests * 2), dpi=512)
    for test_image, reconstructed_image in zip(test_images, reconstructions):
        test_image = test_image.squeeze()
        reconstructed_image = reconstructed_image[:, :, 0]
        plt.subplot(num_tests, 2, 2 * i + 1, )
        plt.imshow(test_image, cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(num_tests, 2, 2 * i + 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f"Reconstructed (SSIM:{structural_similarity(test_image, reconstructed_image, data_range=test_image.max() - test_image.min()):.2f})")

        plt.axis("off")

        i += 1

    plt.savefig(f"out/{time}/reconstruction.png")
    plt.close()


def get_structural_similarity(model, dataset):
    sample_size = 10
    similarity_scores = []
    idx = np.random.choice(len(dataset), sample_size)
    test_images = dataset[idx]
    reconstructions_test = model.predict(test_images)

    for i in range(reconstructions_test.shape[0]):
        original = test_images[i, :, :, 0]
        reconstructed = reconstructions_test[i, :, :, 0]

        similarity_scores.append(
            structural_similarity(original, reconstructed, data_range=original.max() - original.min()))

    average_similarity = np.average(similarity_scores)

    return average_similarity


def train_vq(time, num_embeddings, latent_dim, batch_size):
    (train_data, validate_data, test_data, data_variance) = dataset.oasis_dataset(200)

    vqvae_trainer = Trainer(data_variance, latent_dim=latent_dim, num_embeddings=num_embeddings)
    vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())

    history = vqvae_trainer.fit(train_data, epochs=10, batch_size=batch_size)

    vqvae_trainer.vqvae.save(f"out/{time}/vqvae_model")
    vqvae_trainer.vqvae.save_weights(f"out/{time}/vqvae_model_weights.h5")

    print(f"#################\n\rSSIM:{get_structural_similarity(vqvae_trainer.vqvae, test_data)}\n\r#################")

    plot_reconstructions(vqvae_trainer.vqvae, test_data, time)
    plot_losses(history, time, "vq_vae")


def train_pixel(time, num_embeddings, latent_dim, batch_size, vqvae_train_path):
    (train_data, validate_data, test_data, data_variance) = dataset.oasis_dataset(200)
    vqvae = tf.keras.models.load_model(vqvae_train_path)
    embeddings = vqvae.get_layer("vector_quantizer").embeddings

    encoded_outputs = vqvae.get_layer("encoder").predict(test_data)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

    codebook_indices = tf.argmin((tf.reduce_sum(flat_enc_outputs ** 2, axis=1, keepdims=True)
                                  + tf.reduce_sum(embeddings ** 2, axis=0) - 2
                                  * tf.matmul(flat_enc_outputs, embeddings)), axis=1).numpy().reshape(encoded_outputs.shape[:-1])

    pixel_cnn = modules.PixelCNN(latent_dim, (64, 64), num_embeddings, 2, 2)
    pixel_cnn.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
    history = pixel_cnn.fit(x=codebook_indices, y=codebook_indices, batch_size=batch_size,
                            epochs=30, validation_split=0.2)

    plot_losses(history, time, "pixelcnn")
    pixel_cnn.save(f"out/{time}/pixelcnn_model")
    pixel_cnn.save_weights(f"out/{time}/pixelcnn_model_weights.h5")


def main():
    time = datetime.now().strftime('%H:%M:%S')
    os.mkdir(f"out/{time}")

    num_embeddings = 128
    latent_dim = 16
    batch_size = 4

    train_vq(time, num_embeddings, latent_dim, batch_size)
    train_pixel(time, num_embeddings, latent_dim, batch_size, "samples/vqvae_model")


if __name__ == "__main__":
    main()
