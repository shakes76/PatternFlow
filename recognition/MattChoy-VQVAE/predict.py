"""
Load the VQ-VAE and PixelCNN models, and use that to generate some images.
"""
import tensorflow as tf
vqvae = tf.keras.models.load_model("./vqvae")
pixel_cnn = tf.keras.models.load_model("./pixel_cnn")

def show_generated_images(n_images, priors, generated):
    """
    Create and save an image containing a number of priors and generated images
    """
    for i in range(n_images):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i], cmap="gray")
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated[i].squeeze(), vmin=0, vmax=1, cmap="gray")
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()

# Generate the images
n_priors = n_images
codes = tf.Variable(tf.zeros(shape=(n_priors,) + (None, 16, 16)[1:], dtype=tf.int32))

_, rows, cols = codes.shape

for row in range(rows):
    for col in range(cols):
        print(f"\rrow: {row}, col: {col}", end="")
        dist = tfp.distributions.Categorical(logits=pixel_cnn(codes, training=False))
        probs = dist.sample()

        codes = codes[:, row, col].assign(probs[:, row, col])

quantiser = vqvae.get_layer("quantiser")

embeddings = quantiser.embeddings
codes = tf.cast(codes, tf.int32)
priors_one_hot = tf.one_hot(codes, vqvae.num_embeddings)
priors_one_hot = tf.cast(priors_one_hot, tf.float32)
quantised = tf.matmul(priors_one_hot, embeddings, transpose_b=True)
quantised = tf.reshape(quantised, (-1, *(output_shape[1:])))

# Generate novel images.
decoder = vqvae.get_layer("decoder_1")
generated_samples = decoder.predict(quantised)

show_generated_images(10, codes, generated_samples)
