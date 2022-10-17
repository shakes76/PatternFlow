from os import sys
from os.path import isfile, join
from tensorflow import keras
import train
import tensorflow as tf
from dataset import get_image_slices
from keras.callbacks import History
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

def predict_model_reconstructions(vqvae_trainer, quantizer, priors, encoded_outputs, number_of_reconstructions):
    #TODO predict model
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
    quantized = tf.matmul(priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))
    decoder = vqvae_trainer.vqvae.get_layer("decoder")
    generated_samples = decoder.predict(quantized)

    for i in range(number_of_reconstructions):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i,:,:])
        plt.title("Code_image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i][:, :, 0])
        plt.title("Generated Sample_image")
        plt.axis("off")
        plt.show()
    print("predict with specific model")

def determine_SSIM(test_images_np, trained_vqvae_model, number_of_reconstructions):
    # Compute the Structural Similarity Index (SSIM) between the two images for all test images
    total_score = 0

    for i in range(number_of_reconstructions): #544 is the number of test images
        test_images = test_images_np
        reconstructions_test = trained_vqvae_model.predict(test_images)
        (score, diff) = structural_similarity(test_images[i], reconstructions_test[i], full=True, multichannel=True)
        diff = (diff * 255).astype("uint8")
        total_score += score

    total_score = total_score / number_of_reconstructions #Get the mean SSIM from all test images
    print("Mean SSIM: {}".format(total_score))

def plot_losses(vqvaeTrainer):
    x = 1

def main():
    train_images, test_images, validate_images = get_image_slices()
    if (len(sys.argv) == 3 and sys.argv[1] == "-m"):
        print(sys.argv)
        try:
            #TODO load specific model and predict/generate images with loaded model
            print("load loading VQVAE model and predict/generate images with default PixelCNN")
            vqvae_trainer = keras.models.laod_model(sys.argv[1])
            vqvae_trainer.fit(train_images, epochs=10, batch_size=32)


            
            encoder = vqvae_trainer.vqvae.get_layer("encoder")
            quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

            pixel_cnn, sampler = train.construct_and_train_pixelCNN(encoder, quantizer, vqvae_trainer, train_images)

            encoded_outputs = encoder.predict(test_images)
            flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
            codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
            codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

            priors = train.generate_probabilities_for_samples(pixel_cnn, sampler)
            reconstructions_test = predict_model_reconstructions(vqvae_trainer, quantizer, priors, encoded_outputs, 10)
            determine_SSIM(test_images, vqvae_trainer.vqvae, len(test_images))
            
        except:
            #TODO  Load default model as regular model was able to be predicted with
            ##Check for VQVAE Model in folder
            print('Error occured during loading custom model, default model will be loaded instead')

            encoder = vqvae_trainer.vqvae.get_layer("encoder")
            quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

            vqvae_trainer, quantizer, priors, encoded_outputs, pixel_cnn, sampler = train.main()
            predict_model_reconstructions(vqvae_trainer, quantizer, priors, encoded_outputs, 10)
            determine_SSIM(test_images, vqvae_trainer.vqvae, len(test_images))

    elif (len(sys.argv == 1)):
        print("loading DEFAULT model and predict/generate images with loaded model")
        vqvae_trainer, quantizer, priors, encoded_outputs, pixel_cnn, sampler = train.main()
    else:
        print("$ python3 predict.py [-m <PathToPreTrainedModel>]")

if __name__ == "__main__":
    main()