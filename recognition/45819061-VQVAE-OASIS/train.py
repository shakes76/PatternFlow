from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import BATCH_SIZE, get_data
from modules import VQVAE, PixelCNN, get_pixelcnn


class VQVAETrainer (tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128):
        super(VQVAETrainer, self).__init__()
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.model = VQVAE(self.latent_dim, self.num_embeddings, (256, 256, 1), residual_hiddens=16)
        self.total_loss_tracker = tf.keras.metrics.Mean(namsamee='total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss_tracker = tf.keras.metrics.Mean(name='vq_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        
        with tf.GradientTape() as tape:
            reconstructions = self.model(x)

            reconstruction_loss = (tf.reduce_mean((x - reconstructions)**2)/self.train_variance)
            total_loss = reconstruction_loss + sum(self.model.losses)

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.model.losses))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
            "ssim": tf.image.ssim(x, reconstructions, max_val=1.0)
        }
    
    def test_step(self, x):
        x, _ = x
        reconstructions = self.model(x, training=False)
        return {
            "ssim": tf.image.ssim(x, reconstructions, max_val=1.0)
        }



x_train, x_test, x_validate, mean, data_variance = get_data(0)

data_variance = np.var(x_train / 255.0)
LATENT_DIM = 8
NUM_EMBEDDINGS = 16
vqvae_trainer = VQVAETrainer(data_variance, LATENT_DIM, NUM_EMBEDDINGS)
vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = vqvae_trainer.fit(
    x=x_train, 
    epochs=20, 
    batch_size=BATCH_SIZE, 
    use_multiprocessing=True, 
    validation_data=(x_validate, x_validate), 
    shuffle=True, 
    validation_freq=1
)

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['vqvae_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim((0, 5000))
plt.legend(['total loss', 'reconstruction loss',  'vqvae loss'])
plt.savefig('losses')
plt.close()

plt.plot(history.history['ssim'])
plt.plot(history.history['val_ssim'])
plt.title('Model SSIM')
plt.ylabel('ssim')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'])
plt.savefig('ssim')
plt.close()


vqvae_trainer.model.save('mymodel')

def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()


trained_vqvae_model = vqvae_trainer.model
idx = np.random.choice(len(x_test), 10)
test_images = x_test[idx]
reconstructions_test = trained_vqvae_model.predict(test_images)

for test_image, reconstructed_image in zip(test_images, reconstructions_test):
    show_subplot(test_image, reconstructed_image)

encoder = vqvae_trainer.model.get_layer("encoder")
quantizer = vqvae_trainer.model.get_layer("vector_quantizer")

encoded_outputs = encoder.predict(test_images)
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

for i in range(len(test_images)):
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i].squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(codebook_indices[i])
    plt.title("Code")
    plt.axis("off")
    plt.show()

encoded_training = encoder.predict(x_train)
flat_enc_training = encoded_training.reshape(-1, encoded_training.shape[-1])
codebook_indices_training = quantizer.get_code_indices(flat_enc_training)
codebook_indices_training = codebook_indices_training.numpy().reshape(encoded_training.shape[:-1])

encoded_validation = encoder.predict(x_validate)
flat_enc_validation = encoded_validation.reshape(-1, encoded_validation.shape[-1])
codebook_indices_validation = quantizer.get_code_indices(flat_enc_validation)
codebook_indices_validation = codebook_indices_validation.numpy().reshape(encoded_validation.shape[:-1])


pixelcnn = get_pixelcnn(num_embeddings=NUM_EMBEDDINGS)
pixelcnn.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
pixelcnn.fit(x=codebook_indices_training, y=codebook_indices_training, batch_size=BATCH_SIZE, epochs=30, validation_data=(codebook_indices_validation, codebook_indices_validation))


