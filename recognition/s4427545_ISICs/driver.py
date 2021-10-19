from yolo import *
import sys
import os
# Uses https://github.com/ultralytics/yolov5

def main(arg):
    # Firstly, call trainer and begin training
    batch_size = 40 # from yolo github
    if len(arg) > 1:
        dir = arg[1]
        batch_size = int(arg[2])
    yolo = YOLO(dir, batch_size, valid_split)
    yolo.display_sample()
    #yolo.train()

    # print graphs here

if __name__ == '__main__':
    main(sys.argv)