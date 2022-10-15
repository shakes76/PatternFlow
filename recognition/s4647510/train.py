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
vqvae_trainer.fit(train, epochs=3000, batch_size=8)

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
