"""
global settings
"""
RANDOM_STATE = 7
TEST_SIZE = 0.2
N_FOLDS = 5
FOLD = 0  # Cross Valid

IMG_WIDTH = 512
IMG_HEIGHT = 384
IMG_CHANNELS = 1
SEG_IMG_CHANNELS = 2
BATCH_SIZE = 2

TRAIN_DATA_PATH = 'ISIC2018/'

IMG_PATH = 'ISIC2018_Task1-2_Training_Input_x2'
SEG_PATH = 'ISIC2018_Task1_Training_GroundTruth_x2'

data_gen_args = dict(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2
)
