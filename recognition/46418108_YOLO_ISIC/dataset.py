import os

def move_data_file():
    """
    This function moves the data.yaml file in the Data folder to yolov5
    folder. This file specifies where the training, testing and validating images
    and labels are. 
    """
    os.rename("Data/data.yaml", "yolov5/data/data.yaml")

