import matplotlib.pyplot as plt
import numpy as np
from dataset import *
from modules import *
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

#Writing a wrapper function
def codebook_wrapper_fn(encoder, embeddings):
    """
    The function encodes the images into codebook indices.

    param:
    encoder - Trained encoder from the VQVAE model
    embeddings - Trained latent codebook layer from VQVAE model

    Returns: Returns a mapper function so it can be applied to a
    dataset loader, so that all codebooks do not need to be stored in memory for training
    """

    def mapper(x):
        #Get the encoded outputs and flatten them
        encoded_outputs = encoder(x)

        flat_enc_outputs = tf.reshape(
            encoded_outputs, [-1, tf.shape(encoded_outputs)[-1]])

        codebook_indices = embeddings.get_closest_index(flat_enc_outputs)
        #Find nearest codebook indices
        codebook_indices = tf.reshape(
            codebook_indices, tf.shape(encoded_outputs)[:-1])

        return codebook_indices

    return mapper

def find_data_var(train_ds):
    """
    For the VQVAE model, the variance of the training dataset is required. Calculates the variance of the
    training dataset

    param:
    train_ds - Data loader for the training set

    Returns: Variance of the training dataset
    """
    # Find the mean of the data
    data_mean = 0
    number_data = 0
    for batch in train_ds:
        number_data += len(batch)
        data_mean += tf.reduce_sum(batch)

    data_mean = data_mean / (number_data * img_height * img_width * 3)

    data_var = 0
    # Find the variance of the data
    for batch in train_ds:
        data_var += tf.reduce_sum((batch - data_mean) ** 2)

    data_var = data_var / ((number_data * img_height * img_width * 3) - 1)
    return data_var

def training_plots(history):
    """
    Plots the training error and mean ssim of the validation data for the VQVAE model

    param:
    - History of the error in training
    """
    plt.plot(history.history['reconstruction_loss'], label="train")
    plt.plot(history.history['val_reconstruction_loss'], label="val")
    plt.title('Training Reconstruction Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(history.history['val_mean ssim'], label="val")
    plt.title('Mean SSIM')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

def pixel_cnn_training_plots(history2):
    """
    Plots the training error for the PixelCNN model

    param:
    - History of the error in training
    """
    plt.plot(history2.history['loss'], label="train")
    plt.plot(history2.history['val_loss'], label="val")
    plt.title('PixelCNN Sparse Categorical CrossEntropy Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

#Constants / Hyperparameters
val_split = 0.2
img_height = 256
img_width = 256
batch_size = 32
latent_dim = 128 #Keep larger for more dimensions in the latent space
num_embeddings= 32 #Keep low so it is easier to train the pixelCNN

train_path = "AD_NC/train" #Path to the training data directory
test_path = "AD_NC/test" #Path to the test data directory
#Oasis dataset path
#train_path = "keras_png_slices_data/keras_png_slices_train"
#test_path = "keras_png_slices_data/keras_png_slices_test"

#Load in the training, validation and test data
train_ds = load_train_data(train_path, img_height, img_width, batch_size, val_split)
val_ds = load_validation_data(train_path, img_height, img_width, batch_size, val_split)
test_ds = load_test_data(test_path, img_height, img_width, batch_size)

data_var = find_data_var(train_ds) #Find the variance of the training data

#Initialising and compiling the model
vqvae_model = VQVAEModel(img_shape = (img_height, img_width, 3), embedding_num=num_embeddings, embedding_dim=latent_dim, beta=0.25, data_variance=data_var)
vqvae_model.compile(optimizer=keras.optimizers.Adam())
device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) else '/CPU:0'

#Training the model
with tf.device("/GPU:0"):
  history = vqvae_model.fit(train_ds, epochs = 10, validation_data = val_ds, batch_size = batch_size)

#Saving the weights. Put the path of where you want to store the weights as input
vqvae_model.save_weights("path to store weights")

#Plotting code and results
training_plots(history)

#Plot some reconstructed images from the vqvae model
reconstucted_image(test_ds, vqvae_model)

# Find the average SSIM of Test Data
mean_ssim = test_mean_ssim(test_ds, vqvae_model)
print(mean_ssim)

# Training the PixelCNN.
#Load in the data
test_ds = load_test_data(test_path, img_height, img_width, batch_size)
#Write a wrapper function for the codebook indices for training
codebook_mapper = codebook_wrapper_fn(
        vqvae_model.get_encoder(),
        vqvae_model.get_vq())
codebook_dataset = test_ds.map(codebook_mapper)
pixelcnn_input_shape = vqvae_model.get_encoder().output.shape[1:3]

#Make and compile the model
pixelcnn_model = PixelCNNModel(pixelcnn_input_shape, vqvae_model._embedding_num, 128, 2,2)
pixelcnn_model.compile(optimizer=keras.optimizers.Adam(0.0003),
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       #loss = tf.keras.losses.MeanSquaredError,
                       metrics=["accuracy"],
)

#Training the model
with tf.device("/GPU:0"):
  history2 = pixelcnn_model.fit(codebook_dataset, batch_size=64,epochs=100, validation_data = codebook_val_dataset)

#Saving the weights. Put the path of where you want to store the weights as input
pixelcnn_model.save_weights("path to store weights")

#Training error plots
pixel_cnn_training_plots(history2)
batch=10
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