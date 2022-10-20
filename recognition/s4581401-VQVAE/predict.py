from dataset import *
from modules import *
from train import *
import tensorflow_probability as tfp


def reconstucted_image(test_ds, vqvae_model):
    """
    Plots the original image, codebook for the given image, and decoded image from the VQVAE model

    param:
    test_ds - Data loader for the test dataset
    vqvae_model - Trained VQVAE Model
    """

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
    """ Calcualtes the mean ssim for all images inside the test set

    param:
    test_ds - The data loader for the test dataset
    vqvae_model - Trained VQVAE Model

    Returns: Mean SSIM of the test set
    """
    with tf.device("/GPU:0"):
        index = 0
        for image in test_ds:
            #Append all the ssim values into one array
            if index == 0:
                predicted_batch = vqvae_model.predict_on_batch(image)
                total_ssim_values = (tf.image.ssim(predicted_batch, image, max_val=1.0)).numpy()
            else:
                predicted_batch = vqvae_model.predict_on_batch(image)
                temp_ssim = (tf.image.ssim(predicted_batch, image, max_val=1.0)).numpy()
                total_ssim_values = np.append(total_ssim_values, temp_ssim)
            index = index + 1
    return np.mean(total_ssim_values)

def generate_priors(pixelcnn_model, batch):
    """
    Generates and returns the priors using the given PixelCNN model.

    param:
    pixelcnn_model - Trained pixelcnn model
    batch - Number of samples to generate

    Returns: Generated codebooks
    """

    # Create an empty array of priors.
    priors = np.zeros(shape=(batch,) + (None, 32, 32)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            logits = pixelcnn_model.predict(priors)
            sampler = tfp.distributions.Categorical(logits)
            probs = sampler.sample()
            priors[:, row, col] = probs[:, row, col]

    return priors
#Constants / Hyperparameters
val_split = 0.2
img_height = 256
img_width = 256
batch_size = 32
latent_dim = 128 #Keep larger for more dimensions in the latent space
num_embeddings= 32 #Keep low so it is easier to train the pixelCNN

#Loading in the testing dataset
test_ds = load_test_data(test_path, img_height, img_width, batch_size)

#Call function to re-instance model. Important that the hyperparmaeters used in the initialisation are the same as the trained model
#Comment out if the trained model is already defined and loaded in

#Loading VQVAE
vqvae_model = VQVAEModel(img_shape=(img_height, img_width, 3), embedding_num=num_embeddings,
                         embedding_dim=latent_dim, beta=0.25,
                         data_variance=0.05)
vqvae_model.load_weights("Placeholder path to trained weights")
#Loading PixelCNN
pixelcnn_model = PixelCNNModel(pixelcnn_input_shape, vqvae_model._embedding_num, 128, 2, 2)
pixelcnn_model.load_weights("drive/MyDrive/trained_pcnn_14th_128dim_32embed")

#Plots of VQVAE reconstructions
#Plot some reconstructed images from the vqvae model
reconstucted_image(test_ds, vqvae_model)

# Find the average SSIM of Test Data
mean_ssim = test_mean_ssim(test_ds, vqvae_model)
print(mean_ssim)

#Plots of generated images from PixelCNN

batch = 10
#Generating new Samples
priors = generate_priors(pixelcnn_model, batch)

priors_one_hot = tf.one_hot(priors.astype("int32"), vqvae_model._embedding_num).numpy()

quantized = tf.matmul(
        priors_one_hot.astype("float32"),
        vqvae_model.get_vq()._embedding,
        transpose_b=True)

quantized = tf.reshape(quantized, (-1, *(vqvae_model.get_encoder().output.shape[1:])))

# pass back into the decoder to make new generated decoded images
decoder = vqvae_model.get_decoder()
generated_samples = decoder.predict(quantized)
for i in range(batch):
  plt.subplot(1, 2, 1)
  plt.imshow(priors[i])
  plt.title("Generated codebook sample")
  plt.axis("off")

  plt.subplot(1, 2, 2)
  plt.imshow(generated_samples[i].squeeze(), cmap="gray")
  plt.title("Decoded sample")
  plt.axis("off")
  plt.show()
