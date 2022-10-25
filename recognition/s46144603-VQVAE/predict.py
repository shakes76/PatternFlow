from train import *

train_vqvae = VQVAETrainer(data_variance, latent_dim=64, num_embeddings=256)
train_vqvae.compile(optimizer=tf.keras.optimizers.Adam(), metrics="loss")
train_vqvae.fit(X, epochs=300, batch_size=2)