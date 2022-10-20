#Imports
from dataset.py import ISIC_Dataset
from modules.py import UNet

#Load Training and Validation Data
train_image_path = "./ISIC-2017_Training_Data"
train_segmentation_path = "./ISIC-2017_Training_Part1_GroundTruth"
valid_image_path = "./ISIC-2017_Validation_Data"
valid_segmentation_path = "./ISIC-2017_Validation_Part1_GroundTruth"

train_dataset = ISIC_Dataset(train_image_path, train_segmentation_path)
train_dataset = ISIC_Dataset(valid_image_path, valid_segmentation_path)

#Load model from file
model = UNet()

#Training Loop that records metrics


#Save Trained weights
