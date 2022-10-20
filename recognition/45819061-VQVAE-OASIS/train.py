from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from modules import VQVAE, get_pixelcnn


class VQVAETrainer (tf.keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, residual_hiddens=256):
        super(VQVAETrainer, self).__init__()
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.model = VQVAE(self.latent_dim, self.num_embeddings, (256, 256, 1), residual_hiddens=residual_hiddens)
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



def train(x_train, x_test, x_validate, epochs=30, batch_size=16, out_dir='vqvae', **kwargs):
    data_variance = np.var(x_train)

    vqvae_trainer = VQVAETrainer(data_variance, **kwargs)
    vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
    history = vqvae_trainer.fit(
        x=x_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        use_multiprocessing=True, 
        validation_data=(x_validate, x_validate), 
        shuffle=True, 
        validation_freq=1
    )

    eval_results = vqvae_trainer.evaluate(x_test, x_test, batch_size=batch_size)
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

    
    plt.plot(history.history['ssim'])
    plt.plot(history.history['val_ssim'])
    plt.title('Model ssim')
    plt.ylabel('ssim')
    plt.xlabel('epoch')
    plt.ylim((0, 1))
    plt.legend(['training set', 'validation set'])
    plt.savefig('ssim')
    plt.close()

    vqvae_trainer.model.summary()
    vqvae_trainer.model.save(out_dir)
    return vqvae_trainer.model


def pixelcnn_train(model, x_train, x_test, x_validate, epochs=30, batch_size=16, out_dir='pixelcnn', **kwargs):
    encoder = model.get_layer("encoder")
    quantizer = model.get_layer("vector_quantizer")

    encoded_training = encoder.predict(x_train)
    flat_enc_training = encoded_training.reshape(-1, encoded_training.shape[-1])
    codebook_indices_training = quantizer.get_code_indices(flat_enc_training)
    codebook_indices_training = codebook_indices_training.numpy().reshape(encoded_training.shape[:-1])

    encoded_validation = encoder.predict(x_validate)
    flat_enc_validation = encoded_validation.reshape(-1, encoded_validation.shape[-1])
    codebook_indices_validation = quantizer.get_code_indices(flat_enc_validation)
    codebook_indices_validation = codebook_indices_validation.numpy().reshape(encoded_validation.shape[:-1])


    pixelcnn = get_pixelcnn(encoded_training.shape[1:-1], **kwargs)
    pixelcnn.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = pixelcnn.fit(
        x=codebook_indices_training, 
        y=codebook_indices_training, 
        batch_size=batch_size, 
        epochs=epochs,
        validation_data=(codebook_indices_validation, codebook_indices_validation)
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim((0, 1))
    plt.legend(['training set', 'validation set'])
    plt.savefig('pcnnacc')
    plt.close()
        
    pixelcnn.save('pixelcnn')
    return pixelcnn