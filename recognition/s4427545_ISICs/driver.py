from yolo import *
import sys

# Uses https://github.com/ultralytics/yolov5
def main(arg):
    batch_size = 40 / 10 # YOLO github recommendation
    if len(arg) > 1:
        dir = arg[1]
        batch_size = int(arg[2])
        mode = arg[3]
    yolo = YOLO(dir, batch_size, valid_split)
    if (mode == 'training'):
        yolo.train()
    else:
        yolo.predict()

if __name__ == '__main__':
    main(sys.argv)