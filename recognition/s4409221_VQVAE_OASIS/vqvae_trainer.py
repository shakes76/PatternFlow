import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

#VQ-VAE Trainer
class VQVAETrainer(keras.models.Model):
    """
    A class which trains the VQ-VAE model.

    Attributes:
    vq_vae: builds the VQ-VAE model for training data.

    """
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        """
        The constructor for VQVAETrainer class.

        Parameters:
          train_variance (float): the variance of the input data.
          latern_dim (int) : number of latent dimensions.
          num_embeddings (int) : number of embeddings in vector quantizer of VQVAE model.
        """
      
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property

    def metrics(self):
        """
          A function to define metrics for model performance.

          Returns:
          total_loss_tracker (float) : returns the total loss of the current state of the model
          reconstruction_loss (float) : returns the reconstruction loss of the current state of the model
          vq_loss_tracker (float) : returns the vq_loss_tracker of the current state of the model
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        """
            A function to train the input data step by step and calculate the metrics accordingly.
            
            Parameters:
            x (array #change) : takes in the input training data 

            Returns:
            total_loss_tracker (float) : returns the total loss of the current state of the model
            reconstruction_loss (float) : returns the reconstruction loss of the current state of the model
            vq_loss_tracker (float) : returns the vq_loss_tracker of the current state of the model

        """

          #change 
          
        with tf.GradientTape() as tape:
                # Outputs from the VQ-VAE.
                # og code delete - reconstructions = self.vqvae(x)
                # x = tf.expand_dims(x, axis=1)

            tmp_result = self.vqvae(x) #change tmp

                #Calculate image difference using SSIM

                # print(x.shape)
                # print(tf.expand_dims(x,-1).shape)
                # print(tmp_result.shape)
                #change
                # print("x",x)
                # print("o/p",tmp_result)
                

            img_diff = 1-(tf.image.ssim(x,tmp_result,1.0))
                
              # print("img",img_diff)


            # Calculate the losses. Mean squared error, change
            reconstruction_loss = (
                tf.reduce_mean((x - tmp_result) ** 2) / self.train_variance
            )
            total_loss = ( img_diff + sum(self.vqvae.losses))

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        #self.optimizer = tf.keras.optimizers.Adam(0.001).minimize((-1*img_diff))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss(SSIM)": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
