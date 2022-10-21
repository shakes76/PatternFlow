from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import dataset
import modules
import matplotlib.pyplot as plt


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
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    create_new_brains()