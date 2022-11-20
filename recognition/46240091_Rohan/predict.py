from dataset import *
from modules import *
from train import *

TRAIN = "./keras_png_slices_data/keras_png_slices_train"
TEST = "./keras_png_slices_data/keras_png_slices_test"

#Loading and preprocesing the data 
xnotscaled = data_loader(TRAIN, scale_flag=False)
x = data_loader(TRAIN,scale_flag=True)
xt = data_loader(TEST, scale_flag=True)

#Variance in training images
data_variance = np.var(xnotscaled / 255.0)
#Training data dims
print("This is training data shape: ", x.shape)


#VQVAE training
vqvae, history = vqvae_training(training_data = x, data_variance = data_variance)
#Plotting the training loss
VQVAE_training_plot(history)


#Plotting the actual images with reconstructed ones along with ssim
print("Showing actual images with reconstructed images")
trained_vqvae_model = vqvae.vqvae1
#Getting first 10 images from test data
test_images = xt[:10]
# Prediction
reconstructions_test = trained_vqvae_model.predict(test_images)
#Plotting the reconstructed images
average_ssim = plot_vqvae_recons(test_images, reconstructions_test)
print("Average SSIM = ", average_ssim)  

#PixelCNN model initialization and training, from https://keras.io/examples/generative/vq_vae/#prepare-data-to-train-the-pixelcnn

#Getting the models
encoder = vqvae.vqvae1.get_layer("encoder")
quantizer = vqvae.vqvae1.get_layer("vector_quantizer")

# Flatten outputs from encoder
encoded_outputs = encoder.predict(x)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

# Generating the codebook indices by passing encoder outputs to vqlayer
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

pixelcnn_input_shape = encoded_outputs.shape[1:-1]
#Making the PixelCNN model
pixelcnn = pcnn_model_maker(pixelcnn_input_shape, vqvae)

#Training the model and plotting the loss
trained_pcnn, pcnn_history = pcnn_training(pixelcnn, codebook_indices)
PCNN_training_plot(pcnn_history)

#Generating new images and plotting them to see their clarity
new_images = generate_new_images(pixelcnn, quantizer, vqvae, encoded_outputs.shape)

i1 ,i3, i5 = xt[:3] 
i2, i4, i6 = new_images[:3]
fig = plt.figure(figsize = (12,12))
count = 0
grid = ImageGrid(fig,111, nrows_ncols=(3,2), axes_pad=0.1)
for ax, im in zip(grid, [i1,i2,i3,i4, i5, i6]):
  if (count%2 != 0): 
    ax.set_title('New Generated Image')
  else:
    ax.set_title('Original test image')
  ax.axis("off")
  ax.imshow(im.squeeze(),  cmap='gray')
  count += 1
plt.show()







