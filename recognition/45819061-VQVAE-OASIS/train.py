from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import BATCH_SIZE, get_data
from modules import VQVAE, get_pixelcnn, get_pixelcnn_sampler


class VQVAETrainer (tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128):
        super(VQVAETrainer, self).__init__()
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.model = VQVAE(self.latent_dim, self.num_embeddings, (256, 256, 1), residual_hiddens=16)
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss_tracker = tf.keras.metrics.Mean(name='vq_loss')
        self.ssim_tracker = tf.keras.metrics.Mean(name='ssim')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.ssim_tracker
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

        ssim = tf.image.ssim(x, reconstructions, max_val=2.0)
        self.ssim_tracker.update_state(tf.reduce_mean(ssim))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
            "ssim": self.ssim_tracker.result()
        }
    
    def test_step(self, x):
        x, _ = x
        reconstructions = self.model(x, training=False)
        ssim = tf.image.ssim(x, reconstructions, max_val=2.0)
        self.ssim_tracker.update_state(tf.reduce_mean(ssim))
        return {
            "ssim": self.ssim_tracker.result()
        }

LATENT_DIM = 8
NUM_EMBEDDINGS = 16

def train(x_train, x_test, x_validate, epochs=30):
    data_variance = np.var(x_train)

    vqvae_trainer = VQVAETrainer(data_variance, LATENT_DIM, NUM_EMBEDDINGS)
    vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
    history = vqvae_trainer.fit(
        x=x_train, 
        epochs=epochs, 
        batch_size=BATCH_SIZE, 
        use_multiprocessing=True, 
        validation_data=(x_validate, x_validate), 
        shuffle=True, 
        validation_freq=1
    )

    eval_results = vqvae_trainer.evaluate(x_test, x_test, batch_size=BATCH_SIZE)
    print("Structured similarity score:", eval_results)


    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['reconstruction_loss'])
    plt.plot(history.history['vqvae_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim((0, 2))
    plt.legend(['total loss', 'reconstruction loss',  'vqvae loss'])
    plt.savefig('losses')
    plt.close()

    

    plt.ylim((0, 1))
    plt.legend(['training set', 'validation set'])
    plt.savefig('ssim')
    plt.close()


    vqvae_trainer.model.save('vqvae')
    return vqvae_trainer.model

def show_subplot(original, reconstructed, i):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")
    plt.savefig('fig'+str(i))
    plt.close()

def demo_model(model, x_test):
    idx = np.random.choice(len(x_test), 10)
    test_images = x_test[idx]
    reconstructions_test = model.predict(test_images)

    for i, (test_image, reconstructed_image) in enumerate(zip(test_images, reconstructions_test)):
        show_subplot(test_image, reconstructed_image, i)

    encoder = model.get_layer("encoder")
    quantizer = model.get_layer("vector_quantizer")

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
        plt.savefig('embedding'+str(i))
        plt.close()

def pixelcnn_train(model, x_train, x_test, x_validate, epochs=30):
    encoder = model.get_layer("encoder")
    quantizer = model.get_layer("vector_quantizer")
    decoder = model.get_layer("decoder")

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
    pixelcnn.fit(
        x=codebook_indices_training, 
        y=codebook_indices_training, 
        batch_size=BATCH_SIZE, 
        epochs=epochs,
        validation_data=(codebook_indices_validation, codebook_indices_validation)
    )


    sampler = get_pixelcnn_sampler(pixelcnn)

    prior_batch_size = 10
    priors = np.zeros(shape=(prior_batch_size,) + pixelcnn.input_shape[1:])
    batch, rows, cols = priors.shape

    for row in range(rows):
        for col in range(cols):
            probs = sampler.predict(priors)
            priors[:, row, col] = probs[:, row, col]

    pretrained_embeddings = quantizer.embeddings
    prior_onehot = tf.one_hot(priors.astype("int32"), NUM_EMBEDDINGS).numpy()
    quantized = tf.matmul(prior_onehot.astype("float32"), pretrained_embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(encoded_training.shape[1:])))

    # Generate novel images.
    generated_samples = decoder.predict(quantized)

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.savefig('gen'+str(i))
        
    pixelcnn.save('pixelcnn')
    return pixelcnn