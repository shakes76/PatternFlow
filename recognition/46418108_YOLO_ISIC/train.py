import os

def install_requirements():
    """
    Installs all the requirements needed to train a model. These
    requirements are in a text file that comes with the YOLOv5
    repositry. 
    """
    os.system("pip install -r yolov5/requirements.txt")

def train(batch_size : int, epochs : int, workers : int, weights : str, name : str):
    """
    Trains a model with given specifics using YOLOv5. Results are saved to yolov5/runs/name.

    Parameters:
        batch_size: The batch size to use for training
        epochs: The number of epcochs to use for training 
        workers: The number of workers to use
        weights: The weights to use e.g. 'yolov5l.pt'
        name: The name of the folder that the data is saved to.

    Returns:
        None
    """
    os.system("cd yolov5")
    command = "python train.py --img 640 --batch " + str(batch_size) + " --epochs " + str(epochs)  + " --data data.yaml --weights " + weights +  " --workers " + str(workers) + " --name " + name
    os.system(command)

