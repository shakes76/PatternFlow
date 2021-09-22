from Mask_R_CNN import *

def main():
    # Firstly, call trainer and begin training
    maskRCNN = MaskRCNN() # pass data generator into constructor
    maskRCNN.train() # should have checkpoint logic in here

    # print graphs here

if __name__ == 'main':
    main()