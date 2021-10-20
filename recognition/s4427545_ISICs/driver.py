from yolo import *
import sys

# Uses https://github.com/ultralytics/yolov5
def main(arg):
    # Default parameters
    batch_size = 40 / 10 # YOLO github recommendation, scaled down for laptop
    epochs = 10
    mode = 'training'
    if len(arg) > 2:
        print('Using default parameters')
        batch_size = int(arg[1])
        mode = arg[2]
        epochs = int(arg[3])
    yolo = YOLO(batch_size, epochs)
    if (mode == 'training'):
        print(f'Beginning training with {epochs} epochs and a batch size of {batch_size}')
        yolo.train()
    else:
        print(f'Beginning {mode} on a batch size of {batch_size}')
        yolo.predict()

if __name__ == '__main__':
    print('Program takes inputs of the form: batch_size, mode (training or predict), epochs')
    main(sys.argv)