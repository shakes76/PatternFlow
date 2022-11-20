# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:11:40 2022

Modules file: Contains helper functions/Classes to be used in conjunction
with main scripts. 

@author: Jacob Barrie: s45800023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import os, os.path
import cv2
import umap
from skimage.metrics import structural_similarity as ssim

### HELPER FUNCTIONS ###

def get_filenames(root_dir):
    """
    Parameters
    ----------
    root_dir (string): Directory containing image files

    Returns
    -------
    List (strings): List containing file names.

    """
    return os.listdir(root_dir)

def plot_sample(data):
        """
        Method to plot a random sample of the images. Typically required only
        as a sanity check.

        """
        sample_idx = random.sample(range(1, len(data)), 3)
        fig = plt.figure()
        
        for i in range(len(sample_idx)):
            ax = plt.subplot(1, 3, i+1)
            plt.tight_layout()
            ax.set_title("Sample #{}".format(i+1))
            ax.axis('off')
            ax.imshow(data[sample_idx[i]])
        plt.show()
        
def check_cuda():
    """
    Method for sanity checking CUDA and setting the device. 

    """
    print("Is device available: ",torch.cuda.is_available())
    print("Current Device: ",torch.cuda.current_device())
    print("Name: ",torch.cuda.get_device_name(0))
    
    device = torch.cuda.device(0)
    return device 

#############################

### Modules ###   
class Hyperparameters():
    """
    Class to initialize hyperparameters for use in training
    """
    def __init__(self):
        self.epochs = 20
        self.batch_size = 32
        self.lr = 0.0002
        self.num_training_updates = 150000
        self.num_hiddens = 128
        self.num_residual_hiddens = 32
        self.embedding_dim = 64
        self.num_embeddings = 512
        self.commitment_cost = 0.25
        self.channels_noise = 100
        self.channels_image = 3
        self.features_d = 128
        self.features_g = 128
        
## VQ-VAE ##
# CITATION: https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=kgrlIKYlrEXl
        
class Residual(nn.Module):
    """
    Class defining a residual connection for the encoder/decoder classes.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, 
                      stride=1, 
                      bias=False)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class ResidualBlock(nn.Module):
    """
    Class defining a residual block to be used in the encoder/decoder classes.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.res1 = Residual(in_channels, num_hiddens, num_residual_hiddens)
        self.res2 = Residual(in_channels, num_hiddens, num_residual_hiddens)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.relu(out)
        return out
        
        
class Encoder(torch.nn.Module):
    """
    Encoder class for VAE. 
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Encoder, self).__init__()
        
        # Conv Block (downsample)
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        
        # Residual blocks
        self.resBlock1 = Residual(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_hiddens=num_residual_hiddens)
        self.resBlock2 = Residual(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_hiddens=num_residual_hiddens)
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu1(out)
        out = self.conv_2(out)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        return out

    
        
class Decoder(torch.nn.Module):
    """
    Decoder class for VAE.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        # Input transform
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        # Residual block
        self.resBlock1 = Residual(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_hiddens=num_residual_hiddens)
        
        # Conv Transpose (upsample)
        self.convT1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self.convT2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.resBlock1(out)
        out = self.convT1(out)
        out = self.relu(out)
        out = self.convT2(out)
        return out

class VectorQuantizer(nn.Module):
    """
    Class implementing the Vector quantiser for the VQVAE
    # CITATION: https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=kgrlIKYlrEXl
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings, encoding_indices
    
   
class VQVAE(nn.Module):
    """
    Class combining sub-modules into the full VQ-VAE model.
    """
    def __init__(self, channels_img, num_hiddens, num_residual_hiddens, 
             num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        
        # Pass input through encoder
        self.encoder = Encoder(channels_img, num_hiddens,
                                num_residual_hiddens)
        
        # Conv layer to transform to correct dims
        self.conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        
        # Pass through quantizer
        self.vq = VectorQuantizer(num_embeddings, 
                                  embedding_dim,
                                  commitment_cost)
        
        # Pass through decoder
        self.decoder = Decoder(embedding_dim,
                               num_hiddens, 
                               num_residual_hiddens)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.conv(out)
        loss, quantized, _, _ = self.vq(out) # extract loss and quantized outputs
        out_reconstruction = self.decoder(quantized) # reconstruct

        return loss, out_reconstruction
    
class VQVAEtrain():
    """
    Class to hold hyperparamters and implement the training method for 
    VQVAE model.
    """
    def __init__(self, Hyperparameters, loader, data):
        self.device = torch.device("cuda")
        self.Hyperparameters = Hyperparameters
        self.VQVAE = VQVAE(self.Hyperparameters.channels_image,
                           self.Hyperparameters.num_hiddens, 
                           self.Hyperparameters.num_residual_hiddens,
                           self.Hyperparameters.num_embeddings,
                           self.Hyperparameters.embedding_dim,
                           self.Hyperparameters.commitment_cost)
        self.loader = loader
        self.data = data
        self.lr = self.Hyperparameters.lr
        self.optimizer = torch.optim.Adam(self.VQVAE.parameters(),
                                          lr=self.lr,
                                          amsgrad=False)
        self.VQVAE.to(self.device)
        
    def train(self, train_res_recon_error):
        data_variance = 0.0338 # variance calculated seperately
        i = 0
        for batch, x in enumerate(self.loader):
            data = x.to(self.device)
            self.optimizer.zero_grad()
            
            # Find training loss and data reconstruction
            vq_loss, data_recon  = self.VQVAE(data)
            
            # Reconstruction loss
            recon_error = F.mse_loss(data_recon, data) / data_variance
            
            # total loss
            loss = recon_error + vq_loss
            loss.backward()
        
            self.optimizer.step()
            
            train_res_recon_error.append(recon_error.item())
        
            if (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print()
            
            i+=1
            

        # Save model
        torch.save(self.VQVAE, "D:/Jacob Barrie/Documents/COMP3710/models/" + "vqvae.txt")

class VQVAEpredict():
    """
    Class to implement prediction methods for VQVAE. We will want to visialize
    the original vs reconstructed images, and view an embedding slice from
    the codebook.
    """
    def __init__(self, VQVAE, test):
        self.VQVAE = VQVAE
        self.test = test
        self.device = torch.device("cuda")
        
    def get_quantized(self, x):
       """
        Method to return quantized data

        Parameters
        ----------
        x : Tensor
            embedding indices.

        Returns
        -------
        quantized data

        """ 
       encoded = x
       # Transform to embedding size
       index = torch.zeros(encoded.shape[0], self.VQVAE.vq._num_embeddings, device=x.device)
       print(encoded.shape)
       print(index.shape)
       
       # Transform for correct dim to matmul (VQVAE embedding space)
       index.scatter_(1, encoded, 1)
       
       quantized = torch.matmul(index, self.VQVAE.vq._embedding.weight).view(1, 64, 64, 64)
       return quantized.permute(0, 3, 1, 2).contiguous()
       
       
    def reconstruction(self):
        self.VQVAE.to(self.device)
        # Obtain an image from the test set
        real_img = next(iter(self.test))
        real = real_img[0]
        real = real.to(self.device)
        
        # Pass image through model
        encoded = self.VQVAE.encoder(real)
        conv = self.VQVAE.conv(encoded)
        conv = conv.view(1, 64, 64, 64)   
        # Obtain quantized outputs
        _, quantized, _, _ = self.VQVAE.vq(conv)
        reconstruction = self.VQVAE.decoder(quantized)
        
        # Plot reals
        grid = torchvision.utils.make_grid(real.cpu())
        grid = grid.detach().numpy()
        plot = plt.imshow(np.transpose(grid, (1,2,0)), interpolation='nearest')
        plt.title("Real")
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        plt.clf()
        
        # Plot reconstruction
        grid = torchvision.utils.make_grid(reconstruction.cpu())
        grid = grid.detach().numpy()
        plot = plt.imshow(np.transpose(grid, (1,2,0)), interpolation='nearest')
        plt.title("Reconstructed")
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        
    def embedding_slice(self):
        test = next(iter(self.test))
        test = test[0]
        test = test.to(self.device)
        
        # Pass image through model
        encoded = self.VQVAE.encoder(test)
        conv =self.VQVAE.conv(encoded)
        conv = conv.view(1, 64, 64, 64)  
        _,quantized,encodings,indices = self.VQVAE.vq(conv)
        decoded = self.VQVAE.decoder(quantized)
        
        # Obtain codebook indices visualization
        idx = indices.view(64,64)
        idx = idx.to('cpu')
        index = idx.detach().numpy()
        
        test = test[0].cpu().detach().numpy()
        
        # Obtain quanitzed data visualization
        quantized = self.get_quantized(indices)
        quantized_decoded = self.VQVAE.decoder(quantized)
        quantized_decoded_idx = quantized_decoded[0] 
        quantized_decoded_idx = quantized_decoded_idx.to('cpu')
        quantized_decoded_idx = quantized_decoded_idx.detach().numpy()
        
        # UMAP projection
        proj = umap.UMAP(n_neighbors=3,
                 min_dist=0.1,
                 metric='cosine').fit_transform(self.VQVAE.vq._embedding.weight.data.cpu())
        
        # Visualize results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Real vs Codebook indice vs Quantized')
        ax1.imshow(test)
        ax2.imshow(index)
        ax3.imshow(quantized_decoded_idx[1])
        
        """
        plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
        plt.title("UMAP Projection of embedding space")
        """
        
## DCGAN ##
## CITATION: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/2.%20DCGAN/train.py
# Tutorial for DCGAN model/training

class Discriminator(nn.Module):
    """
    Class implementing the discriminator for DCGAN
    """
    def __init__(self, channels, features):
        super(Discriminator, self).__init__()
        # 64 x 64 - input
        self.conv1 = nn.Conv2d(channels, features, kernel_size=4, stride=2, padding=1)
        self.batch = nn.BatchNorm2d(features)
        self.leaky1 = nn.LeakyReLU(0.2)
        
        # Block 1 - 32x32 input
        self.conv2 = nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1)
        self.batch1 = nn.BatchNorm2d(features*2)
        self.leaky2 = nn.LeakyReLU(0.2)
        
        # Block 2 - reduce to 1
        self.conv3 = nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(features*4)
        self.leaky3 = nn.LeakyReLU(0.2)
        
        # Block 3 - reduce to 32x32
        self.conv4 = nn.Conv2d(features*4, features*8, kernel_size=4, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(features*8)
        self.leaky4 = nn.LeakyReLU(0.2)
        
        
        # Out - reduces to 1 dimension
        self.out = nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky1(out)
        out = self.batch(out)
        out = self.conv2(out)
        out = self.batch1(out)
        out = self.leaky2(out)
        out = self.conv3(out)
        out = self.batch2(out)
        out = self.leaky3(out)
        out = self.conv4(out)
        out = self.batch3(out)
        out = self.leaky4(out)
        out = self.out(out)
        out = self.sigmoid(out)
        return out
        
class Generator(nn.Module):
    """
    Class implementing the generator for DCGAN
    """
    def __init__(self, channels_noise, channels, features):
        super(Generator, self).__init__()     
        # Block 1 - input noise
        self.convT1 = nn.ConvTranspose2d(channels_noise, features*16, kernel_size=4, stride=2, padding=0)
        self.batch1 = nn.BatchNorm2d(features*16)
        self.relu1 = nn.ReLU()
        
        # Block 2
        self.convT2 = nn.ConvTranspose2d(features*16, features*8, kernel_size=4, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(features*8)
        self.relu2 = nn.ReLU()
        
        # Block 3
        self.convT3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=4, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(features*4)
        self.relu3 = nn.ReLU()
        
        # Block 4
        self.convT4 = nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=2, padding=1)
        self.batch4 = nn.BatchNorm2d(features*2)
        self.relu4 = nn.ReLU()
        
        self.convT5 = nn.ConvTranspose2d(features*2, channels, kernel_size=4, stride=2, padding=1)
        self.tanH = nn.Tanh()
        
    def forward(self, x):
        out = self.convT1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.convT2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.convT3(out)
        out = self.batch3(out)
        out = self.relu3(out)
        out = self.convT4(out)
        out = self.batch4(out)
        out = self.relu4(out)
        out = self.convT5(out)
        out = self.tanH(out)
        return out
        

class trainDCGAN():
    """
    Class to hold hyper-parameters and implement the training process for 
    DCGAN.
    
    Returns
    -------
    None.

    """
    def __init__(self, Discriminator, Generator, trainData):
        """
        Initailize hyper-parameters and pass in models/data.
        
        Parameters
        ----------
        VQVAE : nn.Module
            Trained VQVAE model for creating quantized data
        Discriminator : nn.Module
            Discriminator network
        Generator : nn.Module
            Generator network
        trainData : DataLoader
            data loader containing training data
        Returns
        -------
        None.

        """
        self.Hyperparameters = Hyperparameters()
        self.device = torch.device("cuda")
        self.Discriminator = Discriminator
        self.Generator = Generator
        self.trainData = trainData
        self.epochs = 25
        self.lr = 0.0002
        self.optimizer_g = torch.optim.Adam(self.Generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.Discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

    
    def train(self, dlosses, glosses):
        """
        Training method for dcgan

        Parameters
        ----------
        dlosses : list
            list storing discriminator loss
        glosses : list
            list storing generator loss

        Returns
        -------
        None.

        """
        # Initialize
        self.Discriminator.to(self.device)
        self.Generator.to(self.device)
        loss = nn.BCELoss()
        
        # Create noise and labels
        real_label = 1
        fake_label = 0
        fixed_noise = torch.randn(64, self.Hyperparameters.channels_noise, 1, 1).to(self.device)
        
        for epoch in range(self.epochs):
            print("Epoch: ", epoch + 1)
            for batch, x in enumerate(self.trainData):
                data = x.to(self.device)
                batch_size = data.shape[0]

                # Train Discriminator
                self.Discriminator.zero_grad()
                label = (torch.ones(batch_size)*0.9).to(self.device)
                output = self.Discriminator(data).reshape(-1)
                lossDR = loss(output, label)
                D_x = output.mean().item()
                
                # Generate fake to pass through model
                noise = torch.randn(batch_size, 100,1,1).to(self.device)
                fake = self.Generator(noise)
                label = (torch.ones(batch_size)*0.1).to(self.device)
                output = self.Discriminator(fake.detach()).reshape(-1)
                lossDF = loss(output, label)
                
                # Calculate combined loss
                lossD = lossDR + lossDF
                dlosses.append(lossD)
                lossD.backward()
                self.optimizer_d.step()
                
                # Train generator
                self.Generator.zero_grad()
                label = torch.ones(batch_size).to(self.device)
                output = self.Discriminator(fake).reshape(-1)
                lossG = loss(output, label)
                lossG.backward()
                self.optimizer_g.step()

                glosses.append(lossG)
                if batch % 100 == 0:
                    print("Loss D: ", dlosses[-1], "Loss G: ", glosses[-1])
                    
        # Save models            
        torch.save(self.Discriminator, "D:/Jacob Barrie/Documents/COMP3710/models/" + "discriminator.txt")
        torch.save(self.Generator, "D:/Jacob Barrie/Documents/COMP3710/models/" + "generator.txt")
        
        
        
class generateDCGAN():
    """
    Class to use trained generator to output fake images.
    """
    def __init__(self, Generator, VQVAE):
        """

        Parameters
        ----------
        Generator : Generator
            trained Generator model
        VQVAE : VQVAE
            trained VQVAE model

        Returns
        -------
        None.

        """
        self.device = torch.device("cuda")
        self.Generator = Generator
        self.VQVAE = VQVAE
     
    def get_quantized(self, x):
        # Same as other get quantized, just need to add a dim
        encoded = x.unsqueeze(1)
        index = torch.zeros(encoded.shape[0], self.VQVAE.vq._num_embeddings, device=x.device)
        print(encoded.shape)
        print(index.shape)
        index.scatter_(1, encoded, 1)
        quantized = torch.matmul(index, self.VQVAE.vq._embedding.weight).view(1, 64, 64, 64)
        return quantized.permute(0, 3, 1, 2).contiguous()
       
        
    def gen(self):
        """
        Method to use generator to produce fake codebook indices, which are
        decoded and visualized.
        """
        
        # Generate fake
        noise = torch.randn(1, 100, 1, 1).to(self.device)
        with torch.no_grad():
            fake = self.Generator(noise)
            
        # Extract generated indice
        fake_indice = fake[0][0]
        return_indice = fake_indice
        fake_indice = fake_indice.to('cpu')
        fake_indice = fake_indice.detach().numpy()
        
        # Visualize
        plt.imshow(fake_indice)
        
        # Return fake indice for reconstruction
        return return_indice
    
    def reconstruct(self, indice):  
        """
        Method to use trained VQVAE to quantize and decode the generated 
        codebook indice.
        """
        indice = torch.flatten(indice)
        indice = indice.long()
        
        # quantize and decode codebook indice
        gen = self.get_quantized(indice)
        gen = self.VQVAE.decoder(gen)
        
        # Visualise
        output = gen[0][0]
        return_output = output
        output = output.to('cpu')
        output = output.detach().numpy()
        plt.imshow(output)
        
        # return decoded tensor for use with SSIM
        return return_output
    
    def SSIM(self, fake, root_dir, train_loader):
        """
        Method to iterate over images and compute ssim.

        """
        ssim_max_image = 0
        image_ssim_max = None
        num_ssim_passed = 0
        
        for i in range(len(train_loader)):
            # Iterate through training data set
            img_names = get_filenames(root_dir) 
            img_name = os.path.join(root_dir, img_names[i]) # Finds file path based on index
            image = cv2.imread(img_name) # Reads image
            sample = image
            
            # Send image to tensor for computation
            transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()
                ])
            image = transform(image)
            image = image.unsqueeze(0)
            
            # extract info and detach
            fake = fake.detach()
            real = image[0][0].to("cpu").detach().numpy()
            
            # calculate SSIM 
            ssim_image = ssim(fake, real)

