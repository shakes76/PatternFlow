from modules import get_pixelcnn_sampler
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def show_subplot(original, reconstructed, i):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")
    plt.savefig('fig'+str(i))
    plt.close()

def demo_vqvae(model, x_test):
    idx = np.random.choice(len(x_test), 10)
    test_images = x_test[idx]
    reconstructions_test = model.predict(test_images)

    for i, (test_image, reconstructed_image) in enumerate(zip(test_images, reconstructions_test)):
        show_subplot(test_image, reconstructed_image, i)

    encoder = model.get_layer("encoder")
    quantizer = model.get_layer("vector_quantizer")

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
        plt.savefig('embedding'+str(i))
        plt.close()


def sample_images(vqvae, pixelcnn):
    decoder = vqvae.get_layer('decoder')
    quantizer = vqvae.get_layer('vector_quantizer')
    sampler = get_pixelcnn_sampler(pixelcnn)

    prior_batch_size = 10
    priors = np.zeros(shape=(prior_batch_size,) + pixelcnn.input_shape[1:])
    batch, rows, cols = priors.shape

    for row in range(rows):
        for col in range(cols):
            probs = sampler.predict(priors, verbose=0)
            priors[:, row, col] = probs[:, row, col]

    pretrained_embeddings = quantizer.embeddings
    prior_onehot = tf.one_hot(priors.astype("int32"), vqvae.num_embeddings).numpy()
    quantized = tf.matmul(prior_onehot.astype("float32"), pretrained_embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(vqvae.get_layer('encoder').compute_output_shape((1, 256, 256, 1))[1:])))

    # Generate novel images.
    generated_samples = decoder.predict(quantized)

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.savefig('gen'+str(i))
        plt.close()