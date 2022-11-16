from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
import torch.cuda


device = "cuda" if torch.cuda.is_available() else "cpu"


"""
DataLoader class that converts OASIS brain dataset into a meaningful format for 
training using VQVAE.

Paramaters:
    data_path -> a path to the training, testing
    or validation folder of OASIS brain images
    
    The folder structure should look like this:
        
        folder the path points to is data folder.
        
        Data folder should contain a single folder that contains the images
        
        data folder -> img folder -> images 

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
                                   transforms.Normalize((0.5,), (0.5,)),
                                   transforms.Resize((128,128))])
        data_loader = torchvision.datasets.ImageFolder(root = data_path, 
                                                       transform = trans)
        
        self.data = data_loader
        #self.data = self.data/255.0
        
    def __len__(self):
        return len(self.data)
        
                

    #returns the data_loader object
    def __getitem__(self, idx):
        return self.data[idx]


    

