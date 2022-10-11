import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from dataset import *
from modules import *
import tensorflow_probability as tfp

def reconstucted_image(test_ds, vqvae_model):
    # Plot examples of real, codebook and reconstructed using the test dataset.
    for images in test_ds.take(1):
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
            print("SSIM = ", tf.image.ssim(input_image, reconstructed_image_single, max_val=1.0))
            # print(np.min(input_image), np.max(input_image))
            #  print(np.min(reconstructed_image_single), np.max(reconstructed_image_single))
            # print(np.min(codebook_index[i]), np.max(codebook_index[i]))
            plt.subplot(1, 3, 1)
            plt.imshow(tf.squeeze(input_image), cmap="gray")
            plt.title("Input Image")
            plt.axis("off")

            print(codebook_index[i].shape)
            plt.subplot(1, 3, 2)
            plt.imshow(codebook_index[i])
            plt.title("Codebook Image")
            plt.axis("off")
            plt.subplot(1, 3, 3)

            plt.imshow(tf.squeeze(reconstructed_image[i]), cmap="gray")
            plt.title("Reconstructed Image")
            plt.axis("off")
            plt.show()

def test_mean_ssim(test_ds, vqvae_model):
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
    return np.mean(total_ssim_values)


def generate_priors(pixelcnn_model):
    """
    Generates and returns the priors using the given PixelCNN model.
    """

    # Create an empty array of priors.
    batch = 1
    priors = np.zeros(shape=(batch,) + (None, 64, 64)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            logits = pixelcnn_model.predict(priors)
            sampler = tfp.distributions.Categorical(logits)
            probs = sampler.sample()
            priors[:, row, col] = probs[:, row, col]

    return priors

#Writing a wrapper function
def codebook_wrapper_fn(encoder, embeddings):
    """
    Returns a mapper function handle that can be passed to the dataset.map function.
    This function encodes the images into codebook indices
    """

    def mapper(x):
        encoded_outputs = encoder(x)  # Change this to predict?

        flat_enc_outputs = tf.reshape(
            encoded_outputs, [-1, tf.shape(encoded_outputs)[-1]])

        codebook_indices = embeddings.get_closest_index(flat_enc_outputs)

        codebook_indices = tf.reshape(
            codebook_indices, tf.shape(encoded_outputs)[:-1])

        return codebook_indices

    return mapper

val_split = 0.2
img_height = 256
img_width = 256
batch_size = 32
latent_dim = 32
embedding_num = 128


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
"""
plt.plot(history.history['loss'] ,label="train")
plt.plot(history.history['val_loss'] ,label="val")
plt.title('Training Total Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
"""
plt.plot(history.history['reconstruction_loss'],label="train")
plt.plot(history.history['val_reconstruction_loss'],label="val")
plt.title('Training Reconstruction Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

"""
plt.plot(history.history['vq loss'],label="train")
plt.plot(history.history['val_vq loss'],label="val")
plt.title('Vector Quantized Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
"""

plt.plot(history.history['val_mean ssim'],label="val")
plt.title('Mean SSIM')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

reconstucted_image(test_ds, vqvae_model)
# Find the average SSIM of Test Data
mean_ssim = test_mean_ssim(test_ds, vqvae_model)
print(mean_ssim)

# Running the code for PixelCNN.
#Load in the data
test_ds = load_test_data(test_path, img_height, img_width, batch_size)
#Write a wrapper function for the codebook indices for training
codebook_mapper = codebook_wrapper_fn(
        vqvae_model.get_encoder(),
        vqvae_model.get_vq())
codebook_dataset = test_ds.map(codebook_mapper)
pixelcnn_input_shape = vqvae_model.get_encoder().output.shape[1:3]
#Make and compile the model
pixelcnn_model = PixelCNNModel(pixelcnn_input_shape, vqvae_model._embedding_num, 128, 3, 3)
pixelcnn_model.compile(optimizer=keras.optimizers.Adam(),
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=["accuracy"],
)
pixelcnn_model = PixelCNNModel(pixelcnn_input_shape, vqvae_model._embedding_num, 128, 3, 3)
pixelcnn_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=["accuracy"],
)
with tf.device("/GPU:0"):
  pixelcnn_model.fit(codebook_dataset, batch_size=32,epochs=10)

#Generating Samples
priors = generate_priors(pixelcnn_model)

priors_one_hot = tf.one_hot(priors.astype("int32"), vqvae_model._embedding_num).numpy()

quantized = tf.matmul(
    priors_one_hot.astype("float32"),
    vqvae_model.get_vq()._embedding,
    transpose_b=True)

quantized = tf.reshape(quantized, (-1, *(vqvae_model.get_encoder().output.shape[1:])))

# pass back into the decoder
decoder = vqvae_model.get_decoder()
generated_samples = decoder.predict(quantized)

plt.subplot(1, 2, 1)
plt.imshow(priors[0])
plt.title("Generated codebook sample")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(generated_samples.squeeze(), cmap="gray")
plt.title("Decoded sample")
plt.axis("off")
plt.show()