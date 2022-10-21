__author__ = "James Chen-Smith"

from modules import Hyperparameters
from dataset import DataManager
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from PIL import Image
import os
import torch # Import PyTorch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# %%
class Tester():
    """Tester Class
    """
    def __init__(self):
        self.device = self.init_device() # Initialize device
        self.print_device() # Print selected device.
        self.hp = Hyperparameters() # Initialize hyperparameters
        self.data = DataManager(self.hp.channels_image, self.hp.size_batch_vqvae, shuffle=True)
        self.path_state = "recognition\\46481326\\state"
        try: # Try to load last model state
            self.model_vqvae = torch.load(self.path_state + "\\vqvae.txt")
        except (FileNotFoundError): # If model doesn't exist create model
            print("VQVAE Model does not exist, please train with train.Trainer()")
            
    def get_slice_e(self, model_vqvae, dataloader):
        """Returns a slice from the embedding space

        Args:
            model_vqvae (_type_): VQVAE Model
            dataloader (_type_): Dataloader

        Returns:
            _type_: Slice from the embedding space
        """
        X_real = next(iter(dataloader))
        x_real = X_real[0][0].unsqueeze(0) # Obtain sample from dataloader batch
        x_real = x_real.to(self.device)
        encoded = model_vqvae.encoder(x_real)
        preconv = model_vqvae.preconv(encoded)
        _, quantized_x, index_e, encoded_e = model_vqvae.vq(preconv)
        x_decoded = model_vqvae.decoder(quantized_x)
        slice_e = encoded_e
        slice_e_view = slice_e.view(64, 64)
        slice_e_view = slice_e_view.to("cpu")
        slice_e_np = slice_e_view.detach().numpy()
        x_real = x_real[0][0].cpu().detach().numpy()
        return x_real, slice_e, slice_e_np
    
    def get_generator_dcgan(self):
        """Loads the trained DCGAN Generator

        Returns:
            __type__: Generator mode, none if does not exist 
        """
        try: # Try to load last model state
            return torch.load(self.path_state + "\\generator.txt")
        except (FileNotFoundError): # If model doesn't exist 
            print("DCGAN Generator Model does not exist, please train with train.Trainer()")
            return
    
    def get_slice_e_dcgan(self, model_dcgan_generator):
        """Generates a slice with DCGAN Generator

        Args:
            model_dcgan_generator (_type_): DCGAN Generator Model

        Returns:
            _type_: Slice from the DCGAN Generator
        """
        noise = torch.randn(1, 100, 1, 1).to(self.device)
        with torch.no_grad():
            generator_fake = model_dcgan_generator(noise)
        slice_e_dcgan = generator_fake[0][0]
        slice_e_dcgan_np = slice_e_dcgan.to("cpu").detach().numpy()
        # slice_e_dcgan = torch.flatten(slice_e_dcgan)
        return slice_e_dcgan, slice_e_dcgan_np
        
    def decode_batch(self, model_vqvae, dataloader):
        """Encodes and decoded a batch from dataloader

        Args:
            model_vqvae (_type_): VQVAE Model
            dataloader (_type_): Dataloader

        Returns:
            _type_: Unmodified ground truths
            _type_: Decoded batch from dataloader
        """
        X_real = next(iter(dataloader))
        x_real = X_real[0] # Obtain a batch from the dataloader
        x_real = x_real.to(self.device)
        encoded = model_vqvae.encoder(x_real)
        preconv = model_vqvae.preconv(encoded)
        _, quantized_x, _, _ = model_vqvae.vq(preconv)
        x_decoded = model_vqvae.decoder(quantized_x)
        return x_real, x_decoded
    
    def decode_slice_e(self, model_vqvae, slice_e):
        """Decodes a slice from the embedding space

        Args:
            model_vqvae (_type_): VQVAE Model
            slice_e (_type_): Slice from the embedding space

        Returns:
            _type_: Decoded image as pytorch tensor
            _type_: Decoded image as numpy array
        """
        slice_e_quantized = model_vqvae.vq.get_quantized_x(slice_e)
        slice_e_decoded = model_vqvae.decoder(slice_e_quantized).to("cpu")
        slice_e_decoded_np = slice_e_decoded.detach().numpy()[0][0]
        return slice_e_decoded, slice_e_decoded_np
    
    def decode_slice_e_dcgan(self, model_vqvae, slice_e_dcgan):
        """Decodes a slice from the DCGAN Generator output

        Args:
            model_vqvae (_type_): VQVAE Model
            slice_e_dcgan (_type_): Slice from the DCGAN Generator output

        Returns:
            _type_: Decoded image as pytorch tensor
            _type_: Decoded image as numpy array
        """
        slice_e_dcgan = slice_e_dcgan.long()
        slice_e_dcgan_quantized = model_vqvae.vq.get_quantized_x(slice_e_dcgan)
        slice_e_dcgan_decoded = model_vqvae.decoder(slice_e_dcgan_quantized)
        slice_e_dcgan_decoded_np = slice_e_dcgan_decoded[0][0].to("cpu").detach().numpy()
        return slice_e_dcgan_decoded, slice_e_dcgan_decoded_np
    
    # def decode_dcgan(self, model_vqvae, x):
    #     x = x.view(64, 64).unsqueeze(0).unsqueeze(0)
    #     encoded = model_vqvae.encoder(x)
    #     preconv = model_vqvae.preconv(encoded)
    #     _, quantized_x, _, _ = model_vqvae.vq(preconv)
    #     x_decoded = model_vqvae.decoder(quantized_x)
    #     x_decoded_np = x_decoded[0][0].to("cpu").detach().numpy()
    #     return x_decoded, x_decoded_np
    
    def convert_dcgan_slice_e(self, slice_e_dcgan):
        """Maps from DCGAN Generated slice to embedding space slice

        Args:
            slice_e_dcgan (_type_): DCGAN Generated slice

        Returns:
            _type_: Embedding space mapping as pytorch tensor
            _type_: Embedding space mapping as a numpy array
        """
        slice_e_dcgan = torch.flatten(slice_e_dcgan)
        range_intensity = [70, 358]
        range_intensity = [
            range_intensity[0],
            ((range_intensity[1] - range_intensity[0]) / 2) + range_intensity[0],
            ((range_intensity[1] - range_intensity[0]) / 2) + ((range_intensity[1] - range_intensity[0]) / 4) + range_intensity[0],
            range_intensity[1]
        ]
        slice_e_dcgan_min = torch.min(slice_e_dcgan)
        slice_e_dcgan_max = torch.max(slice_e_dcgan)
        num_slice = len(range_intensity)
        size_slice = (slice_e_dcgan_max - slice_e_dcgan_min) / num_slice
        for slice in range(0, num_slice):
            minimum = slice_e_dcgan_min + slice * size_slice
            slice_e_dcgan[torch.logical_and(minimum <= slice_e_dcgan, slice_e_dcgan <= (minimum+size_slice))] = range_intensity[slice]
        # print(torch.unique(slice_e_dcgan))
        slice_e_dcgan_converted = slice_e_dcgan
        slice_e_dcgan_converted_np = slice_e_dcgan_converted.view(64, 64).to("cpu").detach().numpy()
        return slice_e_dcgan_converted, slice_e_dcgan_converted_np
    
    def view_batch(self, batch, save_path=None, show=True):
        """Visualizes and saves a batch

        Args:
            batch (_type_): A batch of images
            save_path (_type_, optional): Path to save image. Defaults to None.
            show (bool, optional): Open window view or not. Defaults to True.
        """
        image = torchvision.utils.make_grid(batch.cpu())
        image_np = image.numpy()
        figure = plt.imshow(np.transpose(image_np, (1, 2, 0)), interpolation='nearest')
        figure.axes.get_xaxis().set_visible(False)
        figure.axes.get_yaxis().set_visible(False)
        if (save_path != None):
            plt.savefig(save_path)
        if (show):
            plt.show()
        plt.close()
            
    def view_single_compare(self, title, image1, image2, save_path=None, show=True):
        """Visualizes 2 images side by side
        Args:
            title (_type_): Title of the comparison image
            image1 (_type_): Image to show on left
            image2 (_type_): Image to show on right
            save_path (_type_, optional): Path to save image. Defaults to None.
            show (bool, optional): Open window view or not. Defaults to True.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(title)
        ax1.imshow(image1)
        ax2.imshow(image2)
        if (save_path != None):
            plt.savefig(save_path)
        if (show):
            plt.show()
        plt.close()
            
    def view_single(self, image, save_path=None, show=True):
        """Visualize a single image

        Args:
            image (_type_): Image to show
            save_path (_type_, optional): Path to save image. Defaults to None.
            show (bool, optional): Open window view or not. Defaults to True.
        """
        plt.imshow(image)
        if (save_path != None):
            plt.savefig(save_path)
        if (show):
            plt.show()
        plt.close()
        
    def print_ssim(self, slice_e_dcgan_decoded):
        """Computes SSIM, prints information and returns name of image with max structual similarity.

        Args:
            slice_e_dcgan_decoded (_type_): Decoded DCGAN slice

        Returns:
            _type_: Name of image with max structual similarity
        """
        # Load test image for SSIM
        root = "recognition\\46481326\\oasis\\train"
        root_image = os.path.join(root, os.listdir(root)[0])
        folder_image = os.listdir(root_image)
        ssim_max_image = 0
        image_ssim_max = None
        num_ssim_passed = 0
        for name_image in tqdm(folder_image):
            image = os.path.join(root_image, name_image)
            image = Image.open(image).convert('RGB')
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
            image = transform(image)
            image = image.unsqueeze(0)

            fake = slice_e_dcgan_decoded[0][0].to("cpu").detach().numpy()
            real = image[0][0].to("cpu").detach().numpy()
            ssim_image = ssim(fake, real)
            if ssim_image > 0.6:
                num_ssim_passed += 1
            if ssim_image > ssim_max_image:
                ssim_max_image = ssim_image
                name_image_ssim_max = name_image
                image_ssim_max = real
                    
        print(f"Number of Images w/SSIM > 0.6 = [{num_ssim_passed}]")
        print(f"Image w/Max SSIM = [{name_image_ssim_max}], Max SSIM = [{ssim_max_image}]")
        return image_ssim_max
        
    def init_device(self):
        """Initiates and returns the PyTorch device that is being used

        Returns:
            device: The device that is used for computation
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    
    def print_device(self):
        """Prints the PyTorch device that is being used.
        """
        print(f"Device Selected = [{self.device}]")