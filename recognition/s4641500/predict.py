from dataset import *
from modules import *
import numpy.random as r

NUM_IMGS = 8

def show_subplot(initial, reconstr):
    """
    Displays original and reconstructed image and their SSIM.
    Calculates and returns the SSIM between the two images
    """
    
    # find SSIM
    image1 = tf.image.convert_image_dtype(initial, tf.float32)
    image2 = tf.image.convert_image_dtype(reconstr, tf.float32)
    ssim = tf.image.ssim(image1, image2, max_val=1.0)
    plt.suptitle("SSIM: %.2f" %ssim)
   
    plt.subplot(1, 2, 1)
    plt.imshow(initial.squeeze() + 0.5, cmap=plt.cm.gray)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstr.squeeze() + 0.5, cmap=plt.cm.gray)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()
    return ssim 

def reconstruct_images(n):
    # select n random test images
    randx = np.random.choice(len(test_data), n)
    test_imgs = test_data[randx]

    # predictions on test images
    reconstructions_test = trained_vqvae_model.predict(test_images)

    # sum of the SSIM of all resconstructed images
    total_ssim = 0.0

    # visualise
    for image, reconstructed in zip(test_imgs, reconstructions_test):
        ssim = show_subplot(image, reconstructed)
        total_ssim = total_ssim + ssim
    
    return test_imgs

def visualise_codes(test_imgs):
    # visualise the orignal images and their discrete codes
    for i in range(len(test_imgs)):
        plt.subplot(1, 2, 1)
        plt.imshow(test_imgs[i].squeeze() + 0.5, cmap=plt.cm.gray)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices[i], cmap=plt.cm.gray)
        plt.title("Code")
        plt.axis("off")
        plt.show()

def plot_pcnn_loss(pcnn_hist):
    # plot loss for PixelCNN model
    plt.plot(pcnn_hist.history['loss'])
    plt.plot(pcnn_hist.history['val_loss'])
    plt.title('PixelCNN Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_pcnn_acc(pcnn_hist):
    # plot accuracy for PixelCNN model
    plt.plot(pcnn_hist.history['accuracy'])
    plt.plot(pcnn_hist.history['val_accuracy'])
    plt.title('PixelCNN Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def prior_gen(pcnn, batch=10):
  """Creates and returns priors generated from PCNN model."""
  
  priors = np.zeros(shape=(batch,) + (pcnn.input_shape)[1:])
  batch, rows, cols = priors.shape

  # iterate over the priors - must be done one pixel at a time
  for row in range(rows):
      for col in range(cols):
          logits = pcnn.predict(priors)
          sampler = tfp.distributions.Categorical(logits)
          probs = sampler.sample()
          priors[:, row, col] = probs[:, row, col]

  return priors

def show_novel_imgs(priors, vqvae, quant, encoded_outputs):
    # embedding lookup.
    pretrained_embeddings = quant.embeds
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae.embed_n).numpy()
    quantized = tf.matmul(priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

    # generate images
    decoder = vqvae.vqvae.get_layer("decoder")
    generated_samples = decoder.predict(quantized)

    # visualise images
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


def main():
    train_images, test_images, train_data, test_data, variance = load_dataset()
    vqvae = keras.models.load_model("vqvae.h5", custom_objects = {"VQ": VQ})
    pcnn = keras.models.load_model("pcnn.h5", custom_objects = {"PixelConvLayer": PixelConvLayer, "ResBlock": ResBlock})
    image_inds = r.choice(len(test_data), 8)
    images = test_data[image_inds]

    n = NUM_IMGS
    ssim = reconstruct_images(n)
    avg_ssim = ssim / n

    visualise_codes(test_imgs)
    #plot_pcnn_loss(pcnn_hist)
    #plot_pcnn_acc(pcnn_hist)

    # generate the priors 
    priors = generate_priors(pcnn)
    print(f"Prior shape: {priors.shape}")
    
    show_novel_imgs(priors, vqvae, quant, encoded_outputs)

if __name__ == "__main__":
	main()


