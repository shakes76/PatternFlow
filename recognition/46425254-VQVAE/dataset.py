import torch.utils as utils
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
import torch.cuda
import os
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Whichever path the keras_png_slices_data is at, it will be restructured in
order for ImageFolder function to work, which sees each data folder as a 
"class" and since there is only one class, a wrapped folder is needed for each
dataset.

Parameters:
    data_path -> path to the contents of the keras_png_slices_data folder
"""
def reorganize_data(data_path):
    training = "train"
    testing = "test"
    validation = "validate"
    
    training_path = os.path.join(data_path, training)
    os.mkdir(training_path)
    
    testing_path = os.path.join(data_path, testing)
    os.mkdir(testing_path)
    valid_path = os.path.join(data_path, validation)
    os.mkdir(valid_path)
    
    
    
    

"""
DataLoader class that converts OASIS brain dataset into a meaningful format for 
training using VQVAE.

Paramaters:
    data_path -> a path to the training, testing
    or validation folder of OASIS brain images

"""

class DataLoader(Dataset):
    """
    In order to make the dataset useful to pytorch functions, it is needed to
    process the dataset (images) into a readable format
    
    This class paramaterizes the training, testing and validation datasets and
    turns them into tensors
    
    """
    def __init__(self, data_path):
        #Normalise
        trans = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
        data_loader = torchvision.datasets.ImageFolder(root = data_path, transform 
                                        = trans)
        
        self.data = data_loader
        
    def __len__(self):
        return len(self.data)
        
                

    #returns the data_loader object
    def __getitem__(self, idx):
        return self.data[idx]
    
    
    

