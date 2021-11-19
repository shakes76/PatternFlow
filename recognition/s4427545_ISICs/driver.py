import yolov5.train as train
import yolov5.detect as detect
import sys

# Uses https://github.com/ultralytics/yolov5
def main(arg):
    # Default parameters
    batch_size = 40 / 10 # YOLO github recommendation, scaled down for laptop
    epochs = 2
    data = 'isic.yaml'
    model = 'yolov5n.pt'
    mode = 'training'
    if len(arg) > 5:
        print('Using command-line parameters')
        batch_size = int(arg[1])
        mode = arg[2]
        epochs = int(arg[3])
        data = arg[4]
        model = arg[5]
    if (mode == 'training'):
        print(f'Beginning training with {epochs} epochs and a batch size of {batch_size}')
        train.run(img=640,batch=batch_size,epochs=epochs,data=data,weights=model)
    else:
        print(f'Beginning {mode} on a batch size of {batch_size}')
        detect.run(weights=model, imgsz=640, conf_thres=0.5, source='datasets/ISIC/test/images')

if __name__ == '__main__':
    print('Program takes inputs of the form: batch_size, mode (training or predict), epochs, data (.yaml file), model_type (.pt file)')
    main(sys.argv)