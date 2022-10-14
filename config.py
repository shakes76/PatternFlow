# config file


# model params
CHANNELS = 1                                          # 1:graysale 3:rgb
LDIM = 128                                            # laten vector dimemsion
SRES = 4                                              # starting resolution
TRES = 256                                            # target resolution

# training params
BSIZE = (32, 32, 32, 32, 16, 16, 8)                   # batch size of each resolution
FILTERS = (256, 256, 256, 256, 128, 64, 32)           # number of filters of each resolution
EPOCHS = (30, 20, 20, 20, 20, 20, 20)                 # training epochs of each resolution

NSAMPLES = 25                                         # number of output images must be a number with int sqrt
TRAINING_IMAGE_DIR = 'your training Images directory' # training image folder
OUT_ROOT = 'your output root folder'                  # output root folder
