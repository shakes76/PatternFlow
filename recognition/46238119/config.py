import torch

TRAIN_IMG_SIZE = 256
DATA_ROOT = '/content/sample_data/OASIS'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-3
BATCH_SIZES = 16
IMG_DIMENSION = 1
Z_DIM = 256
IMG_RESOLUTION = 256
NUM_WORKERS = 0
PIN_MEMORY = True if device == 'cuda' else False
FIXED_SAMPLE = torch.randn(9, Z_DIM).to(DEVICE)