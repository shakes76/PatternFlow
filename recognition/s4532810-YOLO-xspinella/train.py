import os
import sys

"""
File in which the YOLOv5 training is executed for the ISIC dataset.
This can only be run after dataset.py has been run to 
download/preprocess/arrange the dataset, and a ISIC_dataset.yaml
config file has been made. see README for how to make the yaml.
This will only work when run from:
PatternFlow_LC/recognition/s4532810-YOLO-xspinella
directory.

Training stats are saved to runs/train/exp_

Testing states are saved to runs/val/exp_
"""
def train():
    """
    Simply executes the shell command for the training process
    using all the proper parameters
    """
    ### Run training ###
    num_epochs = input("Please enter desired number of epochs (~400 is good): ")
    os.system(f"python3 yolov5_LC/train.py --img 640 --batch 16 --epochs {num_epochs} --data ISIC_dataset.yaml --weights yolov5s.pt")

def test():
    ### Test on test set ###
    # reccomended weights: yolov5_LC/runs/train/exp2/weights/best.pt
    weights_path = input("Please paste the path to the YOLO weights to use. See train.py for reccomended path: ")
    os.system(f"python3 yolov5_LC/val.py --weights {weights_path} --data yolov5_LC/data/ISIC_test.yaml --img 640")

if __name__ == "__main__":
    mode = input("Please enter desired mode (0 : train and test, 1 : train, 2 : test): ")
    if mode == 0:
        train()
        test()
    elif mode == 1:
        train()
    elif mode == 2:
        test()
    else:
        print("Invalid mode entered.")

