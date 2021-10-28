import tensorflow as tf
from model import VQ_VAE

# Fix memory growth issue encountered when using tensorflow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define directory
train_img_folder = 'D:/keras_png_slices_data_/keras_png_slices_train'
val_img_folder = 'D:/keras_png_slices_data_/keras_png_slices_validate'

# Define parameters for the data loader
batch_size = 64
h = 176
w = 176
train_loader = tf.keras.preprocessing.image_dataset_from_directory(
    train_img_folder,
    label_mode=None,
    seed=123,
    color_mode='grayscale',
    image_size=(h, w),
    shuffle=True,
    batch_size=batch_size
)

# Normalise training data b4 feed it into the model
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255, offset=-0.5)
normalized_train_loader = train_loader.map(lambda x: normalization_layer(x))

# Calculate the mean of the training data
train_mu = 0
n_training_samples = 0
for batch in normalized_train_loader:
    n_training_samples += len(batch)
    train_mu += tf.reduce_sum(batch)
train_mu /= (n_training_samples * h * w)
# Calculate the variance of the training data
train_variance = 0
for batch in normalized_train_loader:
    train_variance += tf.reduce_sum((batch - train_mu) ** 2)
train_variance /= (n_training_samples * h * w) - 1
print('Train variance {}'.format(train_variance))

vq_vae_trainer = VQ_VAE(img_h=h, img_w=w, img_c=1, train_variance=train_variance, embedding_dim=24, n_embeddings=1024, recon_loss_type='MSE', commitment_factor=3)
# Print out the architecture of the VQ VAE
vq_vae_trainer.vq_vae.summary()

val_loader = tf.keras.preprocessing.image_dataset_from_directory(
    val_img_folder,
    label_mode=None,
    seed=123,
    color_mode='grayscale',
    image_size=(h, w)
)
normalized_val_loader = val_loader.map(lambda x: normalization_layer(x))

# Adjust learning rate using Exponential Decay method
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-4,
    decay_steps=5000,
    decay_rate=0.96,
    staircase=False
)

# Define method to save model checkpoints
checkpoint_path = 'vq_vae/cp-{epoch:04d}.ckpt'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_MSE',
    verbose=0, 
    save_weights_only=True
)

#optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vq_vae_trainer.compile(optimizer=optimizer)
vq_vae_trainer.fit(normalized_train_loader, epochs=100, validation_data=normalized_val_loader, callbacks=[checkpoint_callback])
# vq_vae_trainer.vq_vae.save('vq_vae')