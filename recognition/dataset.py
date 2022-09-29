import torch
import os
from PIL import Image
import torchvision.transforms as transform
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, im_size=255):
        """
        Image loader base class
        
        Takes a path to folder of images.
        All images are resized to [im_len, im_len] when an item is requested.
        """
        self.path = path
        self.files = os.listdir(self.path)
        self.len = len(self.files)
        self.im_size = im_size
        self.trf = transform.ToTensor()

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = Image.open(self.path + self.files[index])
        image = image.resize((self.im_size, self.im_size))
        return self.trf(image)  * 2 - 1

class DatasetDiffusion(Dataset):
    def __init__(self, path, im_size=255, timesteps=300, start=0.0001, end=0.05):
        """
        Extends the Dataset class to add noise to the returned image for diffusion models
        """
        super().__init__(path, im_size)
        self.timesteps = timesteps
        self.start = start
        self.end = end
        self.betas = self.lr_schedule() #Calculate betas with lr scheduler
        self.setup() #precalculate alpha and beta values
        self.t = torch.Tensor([0]).type(torch.int64)

    def setup(self):
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def lr_schedule(self):
        """
        Learning rate scheduler for beta values.

        Currently using linear scheduler. Change to something better
        """
        return torch.linspace(self.start, self.end, self.timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def update_timestep(self, t):
        """
        Updates current timestep for noise generation
        """
        self.t = t

    def __getitem__(self, index):
        """
        Override __getitem__ to add noise to image based on stored timestep

        returns noisy image and noise within image
        """
        self.t = torch.randint(0,self.timesteps, (1,))

        image = super().__getitem__(index) * 2 - 1 #Scale image between -1 and 1
        noise = torch.randn_like(image)
        self.sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, self.t, image.shape)
        self.sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, self.t, image.shape)
        return self.sqrt_alphas_cumprod_t * image + self.sqrt_one_minus_alphas_cumprod_t * noise, noise, self.t

    