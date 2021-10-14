
################################## DATASET CONFIG ##############################

PATH = './data/AKOA_Analysis/'
DATASET_PATH = './data/processed/'
FIGURE_LOCATION = './display/figures/'
LOGS_PATH = './logs/'

RANDOM_STATE = 42
TEST_SPLIT = 0.20

IMAGE_SIZES = 228
IMAGE_SIZE = (IMAGE_SIZES, IMAGE_SIZES)
LEFT = 0
RIGHT = 1

UNWANTED_FILES = ['Thumbs.db']

RIGHT_TEXT = '_right'
RIGHT_UNDERSCORE_TEXT = '_r_i_g_h_t'
LEFT_TEXT = '_left'
LEFT_UNDERSCORE_TEXT = '_l_e_f_t'

X_TRAIN = 'x_train'
Y_TRAIN = 'y_train'
X_TEST = 'x_test'
Y_TEST = 'y_test'

SAVE_EXT = '.npy'

AUG_FACTOR = 0.2
AUG_VAL = 0.5
BLUR_AUG = 0.5

################################################################################

########################## TRANSFORMER HYPERPARAMETERS #########################


PATCH_SIZE = 8
PATCHES = (IMAGE_SIZES // PATCH_SIZE) ** 2


################################################################################
