import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
from model import PixelCNN, VQ_VAE
from paths import TRAIN_DATA_PATH, VALIDATE_DATA_PATH, TRAINED_VQ_PATH, PRIOR_CHECKPOINT_PATH
# Fix memory growth issue encountered when using tensorflow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Define parameters for the data loader
batch_size = 64
h = 256
w = 256
train_loader = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DATA_PATH,
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

val_loader = tf.keras.preprocessing.image_dataset_from_directory(
    VALIDATE_DATA_PATH,
    label_mode=None,
    seed=123,
    color_mode='grayscale',
    image_size=(h, w)
)
normalized_val_loader = val_loader.map(lambda x: normalization_layer(x))


# Get the encoder and the quantizer from the trained VQ-VAE
vq_vae_wrapper = VQ_VAE(img_h=h, img_w=w, img_c=1, train_variance=0.0347, embedding_dim=24, n_embeddings=256, recon_loss_type='MSE', commitment_factor=2)
vq_vae_wrapper.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
vq_vae_wrapper.load_weights(TRAINED_VQ_PATH)
encoder = vq_vae_wrapper.vq_vae.get_layer('encoder')
quantizer = vq_vae_wrapper.vq_vae.get_layer('vector_quantizer')
# Initialize the parameters for PixelCNN model
pixelcnn_trainer = PixelCNN(h=32, w=32, embedding_dim=24, 
                                encoder=encoder, quantizer=quantizer, 
                                min_val=0, max_val=255, 
                                num_resnet=1, num_hierarchies=1, resnet_activation='concat_elu',
                                num_filters=32, num_logistic_mix=5, 
                                kernel_size=3, dropout_p=0.5)

# Define method to save model checkpoints
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=PRIOR_CHECKPOINT_PATH,
    monitor='val_entropy_loss',
    verbose=0,
    save_weights_only=True
)


# Adjust learning rate using Exponential Decay method
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# Perform training
pixelcnn_trainer.compile(optimizer=optimizer)
pixelcnn_trainer.fit(normalized_train_loader, epochs=100, validation_data=normalized_val_loader, callbacks=[checkpoint_callback])