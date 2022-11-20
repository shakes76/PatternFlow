# File containing constants so that you only need to modify parameters one place
import torch

ROOT_DIR = ".\\s4646780_OASIS_VQVAE\\keras_png_slices_data"
VQVAE_EPOCHS = 1500
VQVAE_LEARNING_RATE = 1e-3
GAN_EPOCHS = 50
LEARNING_RATE_GAN = 2e-4
GAN_BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
