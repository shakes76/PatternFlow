__author__ = "James Chen-Smith"

# %%
"""Import libraries required for PyTorch"""
from modules import Hyperparameters
from PIL import Image
import os # Import OS
import torch # Import PyTorch
import torchvision # Import PyTorch Vision
import numpy as np

# %%
class DatasetDCGAN(torch.utils.data.Dataset):
    """Custom DataSet Class for DCGAN
    
    Args:
        root (str): Directory of dataset
        model (__type__): Model to reference
        device (__type__): PyTorch device to reference
        transform (__type__): Transform(s) 
    """
    def __init__(self, root, model, device, transform=None):
        self.root = os.path.join(root, os.listdir(root)[0]) # Directory path within directory
        self.model = model
        self.device = device
        self.transform = transform
        self.images = os.listdir(self.root)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.root + "\\" + self.images[index]) # Join directory path with image name
        image = self.transform(image) # Transform image
        image = image.unsqueeze(dim=0)
        image = image.to(self.device)
        encoded = self.model.encoder(image)
        preconv = self.model.preconv(encoded)
        _, _, _, encoded_e = self.model.vq(preconv)
        encoded_e = encoded_e.float().to(self.device)
        encoded_e = encoded_e.view(64, 64)
        encoded_e = torch.stack((encoded_e, encoded_e, encoded_e), 0)
        return encoded_e
        
# %%
class DataManager():
    """DataManager Class
    """
    def __init__(self, channels_image, size_batch, shuffle):
        """Initializer for DataManager Class

        Args:
            channels_image (int): Number of channels of image(s) in dataset, i.e. RGB = 3
            size_batch (int): DataLoader batch size
            shuffle (bool): Shuffle the data (true/false)
        """
        transform_totensor = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Grayscale(),
            ]
        ) # Only transform to tensor
        transform_normalize = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(
                    [0.5 for channel in range(channels_image)], 
                    [0.5 for channel in range(channels_image)]
                )
            ]
        ) # Normalizing transform
        self.dataset = dict()
        self.imagefolder = dict()
        self.imagefolder["X_train"] = self.init_imagefolder("recognition\\46481326\\oasis\\train", transform_totensor)
        self.imagefolder["X_test"] = self.init_imagefolder("recognition\\46481326\\oasis\\test", transform_totensor)
        self.dataloader = dict()
        for name_dataset, dataset in self.dataset.items():
            self.dataloader[name_dataset] = self.init_dataloader(dataset, size_batch, shuffle)
        for name_imagefolder, imagefolder in self.imagefolder.items():
            self.dataloader[name_imagefolder] = self.init_dataloader(imagefolder, size_batch, shuffle)
        
    def init_dataset(self, root, transform):
        """Returns a PyTorch Vision dataset for the DataManager

        Returns:
            dataset (DataSet): PyTorch Vision ImageFolder
        """
        return # Function is disabled (may not be necessary in the future)
        dataset = DataSetCustom(
            root=root, 
            transform=transform
            ) # DataSet loader
        return dataset
    
    def init_imagefolder(self, root, transform):
        """Returns a PyTorch Vision dataset for the DataManager

        Returns:
            imagefolder (ImageFolder): PyTorch Vision ImageFolder
        """
        imagefolder = torchvision.datasets.ImageFolder(
            root=root, 
            transform=transform
            ) # ImageFolder loader
        return imagefolder
    
    def init_dataloader(self, dataset, size_batch, shuffle):
        """Returns a PyTorch DataLoader for the DataManager

        Args:
            dataset (DataSet): PyTorch Vision DataSet or ImageFolder
            size_batch (int): Batch size
            shuffle (bool): Shuffle the data (true/false)
        """
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=size_batch,
            shuffle=shuffle
        )
        return dataloader
    
    def get_dataset(self):
        """Returns the DataManager's PyTorch DataSets
        """
        return self.dataset
    
    def get_imagefolder(self):
        """Returns the DataManager's PyTorch ImageFolders
        """
        return self.imagefolder
    
    def get_dataloader(self):
        """Returns the DataManager's PyTorch DataLoaders
        """
        return self.dataloader
    
        