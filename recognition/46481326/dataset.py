# %%
"""Import libraries required for PyTorch"""
import torch # Import PyTorch
import torchvision as tv # Import PyTorch Vision

# %%
class DataManager():
    """DataManager Class
    """
    def __init__(self, root, channels_image, size_batch, shuffle):
        """Initializer for DataManager Class

        Args:
            root (str): Directory of dataset
            channels_image (int): Number of channels of image(s) in dataset, i.e. RGB = 3
            size_batch (int): DataLoader batch size
            shuffle (bool): Shuffle the data (true/false)
        """
        transform_normalize = tv.transforms.Compose(
            [
                # tv.transforms.Resize((size_image, size_image)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    [0.5 for i in range(channels_image)], 
                    [0.5 for i in range(channels_image)]
                )
            ]
        ) # Normalizing transform
        imagefolder = self.init_imagefolder(root, transform_normalize) # Initialize dataset
        self.dataloader = self.init_dataloader(imagefolder, size_batch, shuffle) # Initialize dataloader
        
    def init_imagefolder(self, root, transform) -> tv.datasets.ImageFolder:
        """Returns a PyTorch Vision dataset for the DataManager

        Returns:
            imagefolder (ImageFolder): PyTorch Vision ImageFolder
        """
        imagefolder = tv.datasets.ImageFolder(
            root=root, 
            transform=transform
            ) # Dataset loader
        return imagefolder
    
    def init_dataloader(self, imagefolder, size_batch, shuffle) -> torch.utils.data.DataLoader:
        """Returns a PyTorch DataLoader for the DataManager

        Args:
            imagefolder (ImageFolder): PyTorch Vision ImageFolder
            size_batch (int): Batch size
            shuffle (bool): Shuffle the data (true/false)
        """
        dataloader = torch.utils.data.DataLoader(
            dataset=imagefolder,
            batch_size=size_batch,
            shuffle=shuffle
        )
        return dataloader
    
    def get_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the DataManager's PyTorch DataLoader
        """
        return self.dataloader
    
        