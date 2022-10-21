from dataset import *
from modules import *

EPOCHS = 1
BATCH_SIZE = 128

train_images, test_images, train_data, test_data, data_variance = load_dataset()
vqvae_trainer = Train_VQVAE(data_variance, dim=16, embed_n=128)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
vqvae_hist = vqvae_trainer.fit(train_data, epochs=EPOCHS, batch_size=128)

encoded_out	= vqvae_trainer.vqvae.get_layer("encoder").predict(test_data)
qtiser = vqvae_trainer.vqvae.get_layer("vector_quantizer")
flat_encs = encoded_out.reshape(-1, encoded_out.shape[-1])
codebooks = qtiser.get_code_indices(flat_encs)
codebooks = codebooks.numpy().reshape(encoded_out.shape[:-1])
pixel_cnn = get_pcnn(vqvae_trainer, encoded_out)
pixel_cnn.compile(optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],)
pcnn_training = pixel_cnn.fit(x = codebooks, y = codebooks, batch_size = 128, epochs = EPOCHS, validation_split = 0.1)
pixel_cnn.save("pcnn.h5")


vqvae = vqvae_trainer.vqvae
vqvae.save("vqvae.h5")
pixel_cnn.save("pcnn.h5")