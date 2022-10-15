import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
import zipfile
import os
import dataset, modules

figure_path = "figures/"

# Load data
train, test, validate = dataset.load_data()
data_variance = np.var(train)

# Train VQVAE
vqvae_trainer = modules.VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_trainer.fit(train, epochs=1000, batch_size=32)

# Plot learning
plt.plot(vqvae_trainer.history.history['reconstruction_loss'], label='reconstruction_loss')
plt.plot(vqvae_trainer.history.history['vqvae_loss'], label = 'vqvae_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig(os.path.join(figure_path, "training_plot"))

# Reconstructions
trained_vqvae_model = vqvae_trainer.vqvae
idx = np.random.choice(len(test), 10)
test_images = test[idx]
reconstructions_test = trained_vqvae_model.predict(test_images)

i = 0
for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    filename = os.path.join(figure_path, "reconstruction_" + str(i) + ".png")
    i += 1
    modules.save_subplot(test_image, reconstructed_image, filename)

# Encoding
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

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
    filename = os.path.join(figure_path, "codebook_" + str(i) + ".png")
    plt.savefig(filename)


# Generate the codebook indices.
encoded_outputs = encoder.predict(train)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices_train = quantizer.get_code_indices(flat_enc_outputs)

encoded_outputs = encoder.predict(validate)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices_validate = quantizer.get_code_indices(flat_enc_outputs)

# PixelCNN
num_residual_blocks = 7
num_pixelcnn_layers = 2
pixelcnn_input_shape = encoded_outputs.shape[1:-1]

pixel_cnn = modules.PixelCNN(num_residual_blocks, num_pixelcnn_layers, num_embeddings=256)
pixel_cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),loss = tf.losses.CategoricalCrossentropy(from_logits = True), metrics=['accuracy'])
pixel_cnn.fit(x = codebook_indices_train, epochs = 600, validation_data = codebook_indices_validate, batch_size = 8, 
                    validation_steps = 1, validation_batch_size = 8, steps_per_epoch = 1)

#get the output shape of the encoder
train_gen_1 = dataset.train_codebook_generator(train, vqvae_trainer, batch_size = 1)
data = next(train_gen_1)
out = pixel_cnn.predict(data[0])

# Create an empty array of priors to generate images.
batch = 5
shape = ((batch,) + out.shape[1:])
priors = tf.zeros(shape = shape)
batch, rows, cols, embedding_count = priors.shape
# Iterate over the priors pixel by pixel.
for row in range(rows):
    for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            x = pixel_cnn(priors, training=False)
            dist = tfp.distributions.Categorical(logits=x)
            sampled = dist.sample()
            sampled = tf.one_hot(sampled,256)
            priors = sampled
print(f"Prior shape: {priors.shape}")

#map the one-hot encodings to actual values
embedding_dim = vqvae_trainer.vq_layer.embedding_dim
pretrained_embeddings = vqvae_trainer.vq_layer.embeddings
pixels = tf.constant(priors, dtype = "float32")

quantized = tf.matmul(pixels, pretrained_embeddings, transpose_b=True)

# Generate images.
decoder = vqvae_trainer.decoder
generated_samples = decoder.predict(quantized)
figs =  ['fig1.png','fig2.png','fig3.png','fig4.png','fig5.png']
for i in range(batch):
    plt.imshow(generated_samples[i])
    plt.axis("off")
    plt.savefig(os.path.join(figure_path, "generated_" + str(i) + ".png"))