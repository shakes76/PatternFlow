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
"""
def train():
    """
    Simply executes the shell command for the training process
    using all the proper parameters
    """
    num_epochs = input("Please enter desired number of epochs: ")
    os.system(f"python3 yolov5_LC/train.py --img 640 --batch 16 --epochs {num_epochs} --data ISIC_dataset.yaml --weights yolov5s.pt")

if __name__ == "__main__":
    train()