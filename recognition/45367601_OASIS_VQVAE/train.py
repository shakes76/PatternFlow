import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from modules import *
from pixel import *

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
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
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)
            # Calculate the losses.
            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }




# =====================================================
def train_vqvae(vqvae_trainer, data, latent_dim=16, num_embeddings=32, num_epochs=20, CHECKPOINT='./model_weights/new_weights', LOAD_WEIGHTS=False):

    history = None
    if LOAD_WEIGHTS:
        vqvae_trainer.load_weights(CHECKPOINT)
    else:
        history = vqvae_trainer.fit(data, epochs=num_epochs)
        vqvae_trainer.save_weights(CHECKPOINT)

    return history



# ==============================
def plot_training(history):
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['reconstruction_loss'])
    plt.title('Reconstruction Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "VQVAE Loss"
    plt.plot(history.history['vqvae_loss'])
    plt.title('VQ VAE loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



# ======================================================

def initialise_pixel(encoder_output_shape, vqvae_trainer):
    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
    pixelcnn_input_shape = encoder_output_shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

    pixel_cnn = PixelCNN(pixelcnn_input_shape, 
                                vqvae_trainer, num_pixelcnn_layers, 
                                num_residual_blocks)


    return pixel_cnn

# ==========================================================
# Generate the codebook indices to train the pixel CNN on
def prepare_encodings(train_np, vqvae_trainer):
    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

    pixel_train = train_np[:2000,:,:,:]
    encoded_outputs = encoder.predict(pixel_train, batch_size=2)

    batched_train = batch_np(encoded_outputs, 32)
    codebook_indices = np.empty(shape=(1,64,64))

    for batch in batched_train:
        print("batch")
        flat_enc_outputs = batch.reshape(-1, batch.shape[-1])
        indices = quantizer.get_code_indices(flat_enc_outputs)
        indices = indices.numpy().reshape(batch.shape[:-1])
        codebook_indices = np.append(codebook_indices, indices, 0)

    return codebook_indices[1:,:,:]

# ============================================================
# Compile and run the pixel CNN model
def train_pixel(pixel_cnn_model, codebook_indices, epochs):
    pixel_cnn_model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # need to fix y
    pixel_cnn_model.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=8,
        epochs=epochs,
        validation_split=0.1,
    )
