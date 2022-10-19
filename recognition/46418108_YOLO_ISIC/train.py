import os
import subprocess
from modules import *

def install_requirements():
    os.system("pip install -r yolov5/requirements.txt")

def train(batch_size : int, epochs : int, workers : int, weights: str, name : str):
    os.system("cd yolov5")
    command = "python train.py --img 640 --batch " + str(batch_size) + " --epochs " + str(epochs)  + " --data data.yaml --weights " + weights +  " --workers " + str(workers) + " --name " + name
    print(command)
    #os.system(command)


if __name__ == '__main__':
    #get_yolo()
    #install_requirements()
    train(16, 300, 24, "yolov5l.pt", "yolo_lesion_det")