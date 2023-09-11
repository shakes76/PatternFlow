from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import dataset
import modules
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


def create_new_brains():
    (train_data, validate_data, test_data, data_variance) = dataset.oasis_dataset(images=10)

    vqvae = modules.VQVAE(16, 128)
    vqvae.load_weights("samples/vqvae_model_weights.h5")

    pixelcnn = modules.PixelCNN(16, 128, 2, 2)
    pixelcnn.load_weights("samples/pixelcnn_model_weights.h5")

    inputs = keras.layers.Input(shape=(64, 64, 16))
    outputs = pixelcnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)

    batch = 10
    rows = 64
    cols = 64
    priors = np.zeros(shape=(batch, rows, cols))

    for row in range(rows):
        for col in range(cols):
            priors[:, row, col] = sampler.predict(priors)[:, row, col]
            print(f"{(row + 1)*(col + 1) + (col)}/{64*64}")

    pretrained_embeddings = vqvae.get_layer("vector_quantizer").embeddings
    one_hot = tf.one_hot(priors.astype("int32"), 128).numpy()
    quantized = tf.reshape(tf.matmul(one_hot.astype("float32"),
                                     pretrained_embeddings, transpose_b=True), (-1, *(64, 64, 16)))

    generated_samples = vqvae.get_layer("decoder").predict(quantized)

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i], cmap='gray')
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5, cmap='gray')
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()


def get_structural_similarity():
    vqvae = modules.VQVAE(16, 128)
    vqvae.load_weights("samples/vqvae_model_weights.h5")
    _, _, test_data, _ = dataset.oasis_dataset(500)

    similarity_scores = []
    reconstructions_test = vqvae.predict(test_data)

    for i in range(reconstructions_test.shape[0]):
        original = test_data[i, :, :, 0]
        reconstructed = reconstructions_test[i, :, :, 0]

        similarity_scores.append(structural_similarity(original, reconstructed,
                                                       data_range=original.max() - original.min()))

    average_similarity = np.average(similarity_scores)

    print(average_similarity)


def plot_reconstructions():
    vqvae = modules.VQVAE(16, 128)
    vqvae.load_weights("samples/vqvae_model_weights.h5")
    _, _, test_data, _ = dataset.oasis_dataset(500)

    num_tests = 4
    test_images = test_data[np.random.choice(len(test_data), num_tests)]
    reconstructions = vqvae.predict(test_images)

    i = 0
    plt.figure(figsize=(num_tests * 2, 4), dpi=512)
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

    plt.show()


if __name__ == "__main__":
    plot_reconstructions()