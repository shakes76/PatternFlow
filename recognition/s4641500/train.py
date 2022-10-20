from dataset import *
from modules import *

EPOCHS = 5


# initialise and train
train_images, test_images, train_data, test_data, data_variance = load_dataset()

vqvae_trainer = Train_VQVAE(data_variance, dim=16, embed_n=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_history = vqvae_trainer.fit(train_data, epochs=EPOCHS, batch_size=128)

#vqvae_trainer.save("vqvae.h5")

# Initialise encoder and quantiser
enc = vqvae_trainer.vqvae.get_layer("encoder")
quant = vqvae_trainer.vqvae.get_layer("vector_quantizer")

# Flatten the encoder outputs.
encoded_outputs = enc.predict(train_data)
# reduce indices because my VRAM is insufficient
encoded_outputs = encoded_outputs[:len(encoded_outputs) // 2]
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

# Generate the codebook indices
codebook_indices = quant.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")


pixel_cnn = get_pixelcnn(vqvae_trainer, encoded_outputs)
pixel_cnn.summary()

# Compile the PixelCNN Model
pixel_cnn.compile(optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],)

# Train the PixelCNN Model
pixelcnn_history = pixel_cnn.fit(x=codebook_indices, y=codebook_indices, 
                  batch_size=128, epochs=EPOCHS, validation_split=0.2,)

pixel_cnn.save("pcnn.h5")

def show_subplot(original, reconstructed):
    """
    Displays original and reconstructed image and their SSIM.
    Calculates and returns the SSIM between the two images
    """
    
    # Calculate SSIM
    image1 = tf.image.convert_image_dtype(original, tf.float32)
    image2 = tf.image.convert_image_dtype(reconstructed, tf.float32)
    ssim = tf.image.ssim(image1, image2, max_val=1.0)
    plt.suptitle("SSIM: %.2f" %ssim)
   
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5, cmap=plt.cm.gray)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5, cmap=plt.cm.gray)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()
    return ssim 

##############################################################################
trained_vqvae_model = vqvae_trainer.vqvae
trained_vqvae_model.save("vqvae.h5")

# Choose 10 random test images
idx = np.random.choice(len(test_data), 10)
test_images = test_data[idx]

# Perform predictions on test images
reconstructions_test = trained_vqvae_model.predict(test_images)

# The sum of the SSIM of all resconstructed images
total_ssim = 0.0

# Visualise reconstructions
for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    ssim = show_subplot(test_image, reconstructed_image)
    total_ssim = total_ssim + ssim


# Visualise the orignal images and their discrete codes
for i in range(len(test_images)):
    
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i].squeeze() + 0.5, cmap=plt.cm.gray)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices[i], cmap=plt.cm.gray)
    plt.title("Code")
    plt.axis("off")
    plt.show()

# Plot loss for PixelCNN model

plt.plot(pixelcnn_history.history['loss'])
plt.plot(pixelcnn_history.history['val_loss'])
plt.title('PixelCNN Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot accuracy for PixelCNN model

plt.plot(pixelcnn_history.history['accuracy'])
plt.plot(pixelcnn_history.history['val_accuracy'])
plt.title('PixelCNN Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


def generate_priors(pixel_cnn):
  """
  Generates and returns the priors using the given PixelCNN model.
  """
  
  # Create an empty array of priors.
  batch = 10
  priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
  batch, rows, cols = priors.shape

  # Iterate over the priors because generation has to be done sequentially pixel by pixel.
  for row in range(rows):
      for col in range(cols):
          logits = pixel_cnn.predict(priors)
          sampler = tfp.distributions.Categorical(logits)
          probs = sampler.sample()
          priors[:, row, col] = probs[:, row, col]

  return priors


  # Generate the priors 
priors = generate_priors(pixel_cnn)
print(f"Prior shape: {priors.shape}")


# Perform an embedding lookup.
pretrained_embeddings = quant.embeds
priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.embed_n).numpy()
quantized = tf.matmul(priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True)
quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

# Generate novel images.
decoder = vqvae_trainer.vqvae.get_layer("decoder")
generated_samples = decoder.predict(quantized)

# Visulaise the novel images generated from discrete codes
for i in range(10):
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i], cmap = plt.cm.gray)
    plt.title("Code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5, cmap = plt.cm.gray)
    plt.title("Generated Sample")
    plt.axis("off")
    plt.show()


