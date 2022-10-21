from modules import *
from mpl_toolkits.axes_grid1 import ImageGrid

VQVAE_EPOCHS = 30
VQVAE_BATCHSIZE = 64
PCNN_EPOCHS = 100
PCNN_BATCHSIZE = 128

def vqvae_training(training_data, data_variance, latent_dims = 16, num_embeddings = 128):
  vqvae = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
  vqvae.compile(optimizer=keras.optimizers.Adam())
  history = vqvae.fit(training_data, epochs=VQVAE_EPOCHS, batch_size=VQVAE_BATCHSIZE)
  return vqvae, history

def pcnn_training(pixel_cnn_model, codebook_indices):
  pixel_cnn_model.compile(optimizer=keras.optimizers.Adam(3e-4),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"],)
  history = pixel_cnn_model.fit(x=codebook_indices, y=codebook_indices, batch_size=PCNN_BATCHSIZE, epochs=PCNN_EPOCHS, validation_split=0.1)
  return pixel_cnn_model, history


def VQVAE_training_plot(vqvae_history):
  plt.plot(vqvae_history.history['reconstruction_loss'])
  plt.plot(vqvae_history.history['total_loss'])
  plt.plot(vqvae_history.history['vqvae_loss'])
  plt.title('VQVAE Training Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Reconstruction Loss', 'Total Loss', 'VQVAE loss'])
  plt.ylim([0, 1])
  plt.show()


def PCNN_training_plot(pcnn_history):
  plt.plot(pcnn_history.history['loss'])
  plt.title('PIXELCNN Training Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Training Loss'])
  plt.show()

def plot_vqvae_recons(original, reconstructed):
  """
  Plots 2 reconstructed images from vqvae along with original images in a grid
  and returns the average ssim
  """
  total = 0
  i1 ,i3 = original[:2] 
  i2, i4 = reconstructed[:2]
  fig = plt.figure(figsize = (8,8))
  count = 0
  grid = ImageGrid(fig, 111, nrows_ncols=(2,2), axes_pad=0.1)
  for ax, im in zip(grid, [i1,i2,i3,i4]):
    if (count%2 != 0): 
      ax.set_title('Reconstructed')
    else:
      ax.set_title('Original test image')

    if (count == 0):
      total += tf.image.ssim(i1, i2, max_val=1)
    elif (count == 2):
      total += tf.image.ssim(i3, i4, max_val=1)
    ax.axis("off")
    ax.imshow(im.squeeze(),  cmap='gray')
    count += 1
  plt.show()
  
  return total/2