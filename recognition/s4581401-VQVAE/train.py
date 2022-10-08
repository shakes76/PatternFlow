import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from dataset import *
from modules import *

val_split = 0.2
img_height = 256
img_width = 256
batch_size = 32
latent_dim = 32
embedding_num = 256


PATH = os.getcwd()
print(PATH)

train_path = PATH + "/ADNI_AD_NC_2D/AD_NC/train"
test_path = PATH + "/ADNI_AD_NC_2D/AD_NC/test"
train_ds = load_train_data(train_path, img_height, img_width, batch_size, val_split)
val_ds = load_validation_data(train_path, img_height, img_width, batch_size, val_split)
test_ds = load_test_data(test_path, img_height, img_width, batch_size)

#Find the mean of the data
data_mean = 0
number_data = 0
for batch in train_ds:
    number_data += len(batch)
    data_mean += tf.reduce_sum(batch)

data_mean = data_mean / (number_data * img_height * img_width * 3)

data_var = 0
#Find the variance of the data
for batch in train_ds:
    data_var += tf.reduce_sum((batch - data_mean)**2)

data_var = data_var / ((number_data * img_height * img_width * 3) - 1)

vqvae_model = VQVAEModel(img_shape = (256, 256, 3), embedding_num = 256, embedding_dim= latent_dim, beta = 0.25, data_variance=0.05)
vqvae_model.compile(optimizer=keras.optimizers.Adam())
device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) else '/CPU:0'
with tf.device("/GPU:0"):
  history = vqvae_model.fit(train_ds, epochs = 10, validation_data = val_ds, batch_size = batch_size)


#Plotting code and results
plt.plot(history.history['loss'])
plt.title('Training Total Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['reconstruction_loss'])
plt.title('Training Reconstruction Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['vq loss'])
plt.title('Vector Quantized Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['mean ssim'])
plt.title('Mean SSIM')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.show()

# Plot examples of real, codebook and reconstructed using the test dataset.
for images in train_ds.take(1):
    reconstructed_image = vqvae_model.predict(images)

    # Getting the codebook images. Need to first get the codebook index
    encoder_output = vqvae_model.get_encoder().predict(images)
    # Once we encode the image, flatten and find closest codebook index.
    flattened_output = encoder_output.reshape(-1, encoder_output.shape[-1])
    codebook_index = vqvae_model.get_vq().get_closest_index(flattened_output)

    # Images become (channel x 64 x 64)
    codebook_index = tf.reshape(codebook_index, encoder_output.shape[:-1])
    for i in range(9):
        input_image = tf.reshape(images[i], (1, 256, 256, 3))

        reconstructed_image_single = tf.reshape(reconstructed_image[i], (1, 256, 256, 3))
        print(tf.image.ssim(input_image, reconstructed_image_single, max_val=1.0))
        # print(np.min(input_image), np.max(input_image))
        #  print(np.min(reconstructed_image_single), np.max(reconstructed_image_single))
        # print(np.min(codebook_index[i]), np.max(codebook_index[i]))
        plt.subplot(1, 3, 1)
        plt.imshow(tf.squeeze(input_image), cmap="gray")
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(codebook_index[i])
        plt.title("Codebook Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(reconstructed_image_single), cmap="gray")
        plt.title("Reconstructed Image")
        plt.axis("off")
        plt.show()

# Find the average SSIM of Test Data
with tf.device("/GPU:0"):
    index = 0
    for image in test_ds:
        if index == 0:
            predicted_batch = vqvae_model.predict_on_batch(image)
            total_ssim_values = (tf.image.ssim(predicted_batch, image, max_val=1.0)).numpy()
        else:
            predicted_batch = vqvae_model.predict_on_batch(image)
            temp_ssim = (tf.image.ssim(predicted_batch, image, max_val=1.0)).numpy()
            total_ssim_values = np.append(total_ssim_values, temp_ssim)
        index = index + 1

print(np.mean(total_ssim_values))