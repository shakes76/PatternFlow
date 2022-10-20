# model params
CHANNELS = 1                                          # 1:graysale 3:rgb
LDIM = 200                                            # laten vector dimemsion
SRES = 4                                              # starting resolution
TRES = 256                                            # target resolution

# training params
BSIZE = (16, 16, 16, 8, 8, 4, 4)                      # batch size of each resolution
FILTERS = (256, 256, 256, 256, 128, 64, 32)           # number of filters of each resolution
STAB = False                                          # whether to stablize after fade in
EPOCHS = {
    0: 10,                                            # ephocs of base model
    1: (15, 10),                                      # ephocs of 8x8 fade in and stabilize
    2: (15, 10),                                      # ephocs of 16x16 fade in and stabilize
    3: (15, 10),                                      # ephocs of 32x32 fade in and stabilize
    4: (15, 10),                                      # ephocs of 64x64 fade in and stabilize
    5: (15, 10),                                      # ephocs of 128x128 fade in and stabilize
    6: (15, 10)                                       # ephocs of 256x256 fade in and stabilize
    }                                                   

NSAMPLES = 25                                         # number of output images must be a number with int sqrt
TRAINING_IMAGE_DIR = 'your training Images directory' # training image folder
OUT_ROOT = 'your output root folder'                  # output root folder
