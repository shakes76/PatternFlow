import tensorflow as tf
from model import VQ_VAE, PixelCNN
from paths import TRAINED_VQ_PATH, TRAINED_PRIOR_PATH, TRAIN_DATA_PATH
import matplotlib.pyplot as plt
# Fix memory growth issue encountered when using tensorflow
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

embedding_dim = 24 # Dimension of embeddings in VQ-VAE

#### Need to specify the data loader process. Otherwise, tensorflow will produce errors when loading the trained model
#### I'm not sure why that happens
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
####

# Get the encoder and the quantizer from the trained VQ-VAE
vq_vae_wrapper = VQ_VAE(img_h=h, img_w=w, img_c=1, train_variance=0.0347, embedding_dim=embedding_dim, n_embeddings=256, recon_loss_type='MSE', commitment_factor=2)
vq_vae_wrapper.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
vq_vae_wrapper.load_weights(TRAINED_VQ_PATH)
# Get the quantizer and decoder parts
quantizer = vq_vae_wrapper.vq_vae.get_layer('vector_quantizer')
decoder = vq_vae_wrapper.vq_vae.get_layer('decoder')

## Load prior of VQ VAE
prior_vq_wrapper = PixelCNN(h=32, w=32, embedding_dim=embedding_dim, 
                            encoder=None, quantizer=None, 
                            min_val=0, max_val=255, 
                            num_resnet=1, num_hierarchies=1, 
                            num_filters=32, num_logistic_mix=5, resnet_activation='concat_elu',
                            kernel_size=3, dropout_p=0.5)
prior_vq_wrapper.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
prior_vq_wrapper.load_weights(TRAINED_PRIOR_PATH)

## Generate images
dist = prior_vq_wrapper.dist
N_SAMPLES = 9
codes = dist.sample(N_SAMPLES) # (N_SAMPLES, 32, 32, 1)
codes = tf.cast(codes, dtype=tf.int32)
codes = tf.reshape(codes, [-1, 1]) # (N_SAMPLES * 32 * 32, 1)
onehot_codes = tf.one_hot(codes, quantizer.n_embeddings) # (N_SAMPLES * 32 * 32, N_EMBEDDINGS)
quantized_imgs = tf.matmul(onehot_codes, quantizer.embeddings) # (N_SAMPLES * 32 * 32, EMBEDDING_DIM)
quantized_imgs = tf.reshape(quantized_imgs, [-1, 32, 32, embedding_dim]) # (N_SAMPLES, 32, 32, EMBEDDING_DIM)
images = decoder(quantized_imgs, training=False)

fig, axs = plt.subplots(N_SAMPLES // 3, 3, figsize=(50, 50))
for i in range(N_SAMPLES // 3):
    axs[i, 0].imshow(images[(i * 3) + 0].numpy(), cmap=plt.cm.gray)
    axs[i, 1].imshow(images[(i * 3) + 1].numpy(), cmap=plt.cm.gray)
    axs[i, 2].imshow(images[(i * 3) + 2].numpy(), cmap=plt.cm.gray)
    axs[i, 0].axis('off')
    axs[i, 1].axis('off')
    axs[i, 2].axis('off')
plt.show()