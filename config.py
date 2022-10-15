# model params
CHANNELS = 1                                          # 1:graysale 3:rgb
LDIM = 128                                            # laten vector dimemsion
SRES = 4                                              # starting resolution
TRES = 256                                            # target resolution

# training params
BSIZE = (32, 32, 32, 32, 16, 16, 8)                   # batch size of each resolution
FILTERS = (256, 256, 256, 256, 128, 64, 32)           # number of filters of each resolution
EPOCHS = {
    0: 50,                                            # ephocs of base model
    1: (20, 10),                                      # ephocs of 8x8 fade in and stabilize
    2: (20, 20),                                      # ephocs of 16x16 fade in and stabilize
    3: (20, 20),                                      # ephocs of 32x32 fade in and stabilize
    4: (30, 20),                                      # ephocs of 64x64 fade in and stabilize
    5: (30, 30),                                      # ephocs of 128x128 fade in and stabilize
    6: (30, 30)                                       # ephocs of 256x256 fade in and stabilize
    }

NSAMPLES = 25                                         # number of output images must be a number with int sqrt
TRAINING_IMAGE_DIR = 'your training Images directory' # training image folder
OUT_ROOT = 'your output root folder'                  # output root folder
