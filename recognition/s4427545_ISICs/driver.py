from Mask_R_CNN import *
import sys
import os
# Uses https://github.com/ahmedfgad/Mask-RCNN-TF2

def main(arg):
    # Firstly, call trainer and begin training
    dir = os.path.expanduser('~/Datasets/ISIC2018_Task1-2_Training_Data/') # default
    os.environ['AUTOGRAPH_VERBOSITY'] = '10'
    valid_split = 0.2
    batch_size = 32
    if len(arg) > 1:
        dir = arg[1]
        batch_size = int(arg[2])
    maskRCNN = MaskRCNN(dir, batch_size, valid_split) # pass data generator into constructor
    maskRCNN.train() # should have checkpoint logic in here

    # print graphs here

if __name__ == '__main__':
    main(sys.argv)