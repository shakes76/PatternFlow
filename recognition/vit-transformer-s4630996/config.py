# data
BATCH_SIZE = 128
IMAGE_SIZE = 200  # We'll resize input images to this size
NUM_CLASS = 2
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
MU = 45
VARIANCE = 60

# patches
PATCH_SIZE = 20  
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# transformer-econder
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 4

# mlp head
MLP_HEAD_UNITS = [256]  # Size of the dense layers of the final classifier

# model
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 2
DROPOUTS = {"mha": 0.2, "encoder_mlp": 0.2, "mlp_head": 0.5}