"""
train.py

Alex Nicholson (45316207)
11/10/2022

Contains the source code for training, validating, testing and saving your model. The model is imported from “modules.py” and the data loader is imported from “dataset.py”. Losses and metrics are plotted throughout training.

"""


import dataset
import modules
import utils
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio.v2 as imageio
import os as os


class VQVAETrainer(keras.models.Model):
    """
    A Custom Training Loop for the VQ-VAE Model

        Attributes:
            train_variance (ndarray): The input data for the model (input data in the form of variances?) ???
            latent_dim (int): The number of latent dimensions the images are compressed down to (default=32)
            num_embeddings (int): The number of codebook vectors in the embedding space (default=128)
            vqvae (Keras Model): The custom VQ-VAE model
            total_loss_tracker (Keras Metric): A tracker for the total loss performance of the model during training???
            reconstruction_loss_tracker (Keras Metric): A tracker for the reconstruction loss performance of the model during training???
            vqvae_loss_tracker (Keras Metric): A tracker for the VQ loss performance of the model during training???

        Methods:
            metrics(): Returns a list of metrics for the total_loss, reconstruction_loss, and vqvae_loss of the model
            train_step(x): Trains the model for a single step using the given training sample/samples x???
    """

    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = modules.get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vqvae_loss_tracker = keras.metrics.Mean(name="vqvae_loss")
        self.ssim_history = []

    @property
    def metrics(self):
        """
        Gets a list of metrics for current total_loss, reconstruction_loss, and vqvae_loss of the model
        
            Returns:
                A list of metrics for total_loss, reconstruction_loss, and vqvae_loss
        """

        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vqvae_loss_tracker,
        ]

    def train_step(self, x):
        """
        Trains the model for a single step using the given training sample/samples x???

            Parameters:
                x (Tensorflow Tensor???): The input training sample/samples (how big is a training step? how many samples?) ???

            Returns:
                A dictionary of the model's training metrics with keys: "loss", "reconstruction_loss", and "vqvae_loss"
        """

        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vqvae_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vqvae_loss_tracker.result(),
        }


class ProgressImagesCallback(keras.callbacks.Callback):
    """
    A custom callback for saving training progeress images
    """

    def __init__(self, train_data, validate_data):
        self.train_data = train_data
        self.validate_data = validate_data

    def save_progress_image(self, epoch):
        """
        Saves progress images as we go throughout training

            Parameters:
                epoch (int): The current training epoch
        """

        num_examples_to_generate = 16
        idx = np.random.choice(len(self.train_data), num_examples_to_generate)
        test_images = self.train_data[idx]
        reconstructions_test = self.model.vqvae.predict(test_images)

        fig = plt.figure(figsize=(16, 16))
        for i in range(reconstructions_test.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(reconstructions_test[i, :, :, 0], cmap='gray')
            plt.axis('off')

        plt.savefig('out/image_at_epoch_{:04d}.png'.format(epoch+1))
        plt.close()

    def create_gif(self):
        """
        Show an animated gif of the progress throughout training
        """

        anim_file = 'out/vqvae_training_progression.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('out/image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                image = imageio.imread(filename)
                writer.append_data(image)

    def on_epoch_end(self, epoch, logs=None):
        self.save_progress_image(epoch)
        
        similarity = utils.get_model_ssim(self.model.vqvae, self.validate_data)
        self.model.ssim_history.append(similarity)
        print(f"ssim: {similarity}")

    def on_train_end(self, logs=None):
        self.create_gif()



def train_vqvae():
    # ---------------------------------------------------------------------------- #
    #                                HYPERPARAMETERS                               #
    # ---------------------------------------------------------------------------- #
    NUM_TRAINING_EXAMPLES = None

    TRAINING_EPOCHS = 20
    BATCH_SIZE = 128

    NUM_LATENT_DIMS = 16
    NUM_EMBEDDINGS = 128

    EXAMPLES_TO_SHOW = 10


    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    (train_data, validate_data, test_data, data_variance) = dataset.load_dataset(max_images=NUM_TRAINING_EXAMPLES, verbose=True)


    # ---------------------------------------------------------------------------- #
    #                                  BUILD MODEL                                 #
    # ---------------------------------------------------------------------------- #
    # Create the model (wrapped in the training class to handle performance metrics logging)
    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=NUM_LATENT_DIMS, num_embeddings=NUM_EMBEDDINGS)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    
    vqvae_trainer.vqvae.summary()

    # ---------------------------------------------------------------------------- #
    #                                 RUN TRAINING                                 #
    # ---------------------------------------------------------------------------- #
    print("Training model...")
    # Run training, plotting losses and metrics throughout
    history = vqvae_trainer.fit(train_data, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, callbacks=[ProgressImagesCallback(train_data, validate_data)])


    # ---------------------------------------------------------------------------- #
    #                                SAVE THE MODEL                                #
    # ---------------------------------------------------------------------------- #
    # Get the trained model
    trained_vqvae_model = vqvae_trainer.vqvae

    # Save the model to file as a tensorflow SavedModel
    trained_vqvae_model.save("./vqvae_saved_model")


    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    # Visualise the model training curves
    utils.plot_training_metrics(history)
    utils.plot_ssim_history(vqvae_trainer.ssim_history)
    
    # Visualise output generations from the finished model
    # utils.show_reconstruction_examples(trained_vqvae_model, validate_data, EXAMPLES_TO_SHOW)




def train_pixelcnn():
    # ---------------------------------------------------------------------------- #
    #                                HYPERPARAMETERS                               #
    # ---------------------------------------------------------------------------- #
    # EXAMPLES_TO_SHOW = 10
    
    NUM_EMBEDDINGS = 128
    NUM_RESIDUAL_BLOCKS = 2
    NUM_PIXELCNN_LAYERS = 2
    
    BATCH_SIZE = 128
    NUM_EPOCHS = 60
    VALIDATION_SPLIT = 0.1

    reuse_codebook_indices = True
    continue_training = False

    # ---------------------------------------------------------------------------- #
    #                                   LOAD DATA                                  #
    # ---------------------------------------------------------------------------- #
    # Import data loader from dataset.py
    (train_data, validate_data, test_data, data_variance) = dataset.load_dataset(max_images=3000, verbose=True)


    # ---------------------------------------------------------------------------- #
    #                          IMPORT TRAINED VQVAE MODEL                          #
    # ---------------------------------------------------------------------------- #
    # Import trained and saved model from file
    trained_vqvae_model = keras.models.load_model("./vqvae_saved_model")

    # ---------------------------------------------------------------------------- #
    #                     GENERATE TRAINING DATA FOR PIXEL CNN                     #
    # ---------------------------------------------------------------------------- #
    print("Generating pixelcnn training data...")
    # Generate the codebook indices.
    encoded_outputs = trained_vqvae_model.get_layer("encoder").predict(train_data)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

    if reuse_codebook_indices == True and os.path.exists('./codebook_indices.csv'):
        print("Loading pre-computed codebook indices from file")
        # Pull the codebook indices from file
        codebook_indices = np.loadtxt('./codebook_indices.csv', delimiter=',')
    else:
        print("Calculating codebook indices")
        # Calculate the codebook indices from scratch
        codebook_indices = utils.get_code_indices_savedmodel(trained_vqvae_model.get_layer("vector_quantizer"), flat_enc_outputs)
        np.savetxt('./codebook_indices.csv', codebook_indices, delimiter=',')

    print("D")

    codebook_indices = codebook_indices.reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")


    # ---------------------------------------------------------------------------- #
    #                                  BUILD MODEL                                 #
    # ---------------------------------------------------------------------------- #
    print("Building model...")
    
    if continue_training:
        # Continue the training of an aoldready part trained model
        pixel_cnn = keras.models.load_model("./pixelcnn_saved_model")
    else:
        # Start training from scratch
        pixelcnn_input_shape = trained_vqvae_model.get_layer("encoder").predict(train_data).shape[1:-1]
        pixel_cnn = modules.get_pixel_cnn(trained_vqvae_model, pixelcnn_input_shape, NUM_EMBEDDINGS, NUM_RESIDUAL_BLOCKS, NUM_PIXELCNN_LAYERS)

    
    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pixel_cnn.summary()

    # ---------------------------------------------------------------------------- #
    #                                 RUN TRAINING                                 #
    # ---------------------------------------------------------------------------- #
    print("Training model...")
    pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )


    # ---------------------------------------------------------------------------- #
    #                                SAVE THE MODEL                                #
    # ---------------------------------------------------------------------------- #
    # Get the trained model
    trained_pixelcnn_model = pixel_cnn

    # Save the model to file as a tensorflow SavedModel
    trained_pixelcnn_model.save("./pixelcnn_saved_model")

    # ---------------------------------------------------------------------------- #
    #                                 FINAL RESULTS                                #
    # ---------------------------------------------------------------------------- #
    # Visualise the discrete codes
    examples_to_show = 10
    utils.visualise_codes(trained_vqvae_model, test_data, examples_to_show)

    # # Visualise novel generations from codes
    num_embeddings = 128
    utils.visualise_codebook_sampling(trained_vqvae_model, pixel_cnn, train_data, num_embeddings, examples_to_show)




if __name__ == "__main__":
    train_vqvae()
    train_pixelcnn()
