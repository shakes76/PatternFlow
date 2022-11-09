from model import VectorQuantizer, createResidualBlock
import tensorflow as tf

# Fix memory growth issue encountered when using tensorflow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Check if the resnet block is runnable
print('Test if the resetnet block is executable and produces the correct output shape')
var = tf.random.uniform(shape=[10, 28, 28, 1])
var_shape = tf.shape(var)
var_output = createResidualBlock(var, n_channels=1, latent_kernel_size=3)
print(var_shape == tf.shape(var_output))


# Test if the quantizer produces the correct output shape
print('Test if the VQ-VAE quantizer is executable and produces the correct output shape')
embedding_dim = 64
n_embeddings = 200
vae_quantizer = VectorQuantizer(n_embeddings, embedding_dim, beta=0.25)
img = tf.ones(shape=[28, 28, embedding_dim])
quantized_img = vae_quantizer(img)
print('Shape of the quantized img: {}'.format(tf.shape(quantized_img)))