"""
This script loads the input and output images from the ISIC dataset for pre-processing.
TODO: Add descriptions of future implementations here.
@author: Mujibul Islam Dipto
"""
import os # for operating system functionalities 


# load data from ISIC dataset 
input_images = sorted(item for item in os.listdir("../../../isic-data/ISIC2018_Task1-2_Training_Input_x2/") if item.endswith('jpg')) # training input
output_images = sorted(item for item in os.listdir("../../../isic-data/ISIC2018_Task1_Training_GroundTruth_x2/") if item.endswith('png')) # ground truth
