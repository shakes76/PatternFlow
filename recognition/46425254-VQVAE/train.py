import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
import numpy as np
import os
import modules
import dataset
import predict

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


NUM_EMBEDDINGS = 64
LATENT_DIM = 16


"""
Class dedicated to training and testing of VQVAE

"""
class VQ_Training():
    
    """
    Initialise VQVAE with following parameters:
        
    Parameters:
        learning_rate -> learning rate of the VQVAE trainer
        epochs -> number of epochs for the model to train
        train_path -> the path of the training data, the path should point to a
        folder containing only the training data folder
        test_path -> the path of the test data, the path should point to a
        folder containing only the testing data folder
        save -> save file for the model if there is any in .pt format, 
        none by default
        visualise -> boolean, plots encoded and decoded images, compared with
        the originals
    """
    def __init__(self, learning_rate, epochs, train_path, test_path, 
                 num_embeddings, latent_dim, save = None, visualise = False, 
                 batch_size = 16):
        super(VQ_Training).__init__()
        # cuda should be used at all times.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        self.train_path = train_path
        self.epochs = epochs
        # use custom dataloader for both train and test sets
        self.data = dataset.DataLoader(train_path)
        self.data2 = dataset.DataLoader(test_path)
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        # batch size of 16
        self.training_data = \
            utils.data.DataLoader(self.data, batch_size = batch_size, \
                                  shuffle = True)
        
        self.testing_data = \
            utils.data.DataLoader(self.data2, batch_size = 1, shuffle = True)
        # check if save file path already exists, if so, reload model from 
        # there
        if save != None and os.path.isfile(save) == True:
            model = modules.VQVAE(self.num_embeddings,self.latent_dim,0.25)
            state_dict = torch.load(save, map_location="cpu")
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
        else:
            self.model = \
                modules.VQVAE(self.num_embeddings,self.latent_dim,0.25
                              ).to(self.device)
        
        self.optimizer = \
            torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        
        self.save = save
        self.visualise = visualise
        
    """
    training of VQVAE
    """
    def train(self):
        epoch = 0
        loss = []
        while epoch != self.epochs:
            
            sub_step = 0
            for i, _ in self.training_data:
                i = i.view(-1, 3, 128, 128).to(self.device)
                
                decoder_outputs, VQ_loss = self.model(i)
                #reset the optimizer gradients to 0 to avoid reusing prev 
                #iteration's gradient
                self.optimizer.zero_grad()
                
                #calculate reconstruction loss
                recon_loss = nn.functional.mse_loss(decoder_outputs, i)
                
                #calculate total loss
                total_loss = recon_loss + VQ_loss
                
                #update the gradient 
                total_loss.backward()
                self.optimizer.step()
                
                if sub_step == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    loss.append(total_loss.item())
                    # save the model every epoch
                    if self.save != None:
        
                        torch.save(self.model.state_dict(), self.save)
                    # visualise if visualise boolean is True and the model 
                    # can be saved
                    if self.visualise == True and self.save != None:
                        predict.VQVAE_predict(self.save, self.num_embeddings, \
                                              self.latent_dim, self.train_path, 
                                              (0,0))       
                sub_step += 1
            epoch += 1
        plt.plot(np.arange(0, self.epochs), loss)
        plt.title("Training Loss vs Number of Epochs")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Training Loss")
        plt.show()
        
                    
        # save anyways if loop finishes by itself
        if self.save != None:
            torch.save(self.model.state_dict(), self.save)
    """
    Testing for VQVAE outputs
    
    Put test data into VQVAE, record Maximum SSIM from all the test data, and 
    avereage SSIM.
    
    """
    def test(self):
        ssim_list = []
        max_ssim = 0
        for i, _ in self.testing_data:
            i = i.view(-1, 3, 128, 128).to(self.device).detach()
            real_grid = torchvision.utils.make_grid(i, normalize = True)
            with torch.no_grad():
                decoded_img , _ = self.model(i)
                decoded_img = \
                    decoded_img.view(-1, 3, 128,128).to(self.device).detach()
                decoded_grid = \
                    torchvision.utils.make_grid(decoded_img, normalize = True)
                decoded_grid = decoded_grid.to("cpu").permute(1,2,0)
                real_grid = real_grid.to("cpu").permute(1,2,0)
                val = ssim(real_grid.numpy(), 
                           decoded_grid.numpy(), channel_axis = -1)
                if max_ssim < val:
                    max_ssim = val
                ssim_list.append(val)
        print("The Average SSIM for the test data is " + 
              (str) (np.average(ssim_list)))
        print("The maximum SSIM is " + (str) (np.amax(ssim_list)))
        
        

"""
Class dedicated to training Pixel CNN model
"""
class PixelCNN_Training():
    """
    Parameters:
        lr -> learning rate of model
        epochs -> number of epochs
        model_path -> path of the VQVAE model needed for the training process
        data_path -> path of the training data
        num_embeddings -> number of embeddings for VQVAE codebook
        latent_dim -> latent dimension of VQVAE codebook
        save -> save file for PixelCNN model if there is any, in .pt format
    """
    def __init__(self, lr, epochs, model_path, data_path, num_embeddings, 
                 latent_dim, save = None):
        
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.epochs = epochs
        self.model_path = model_path
        model = modules.VQVAE(num_embeddings,latent_dim,0.25)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.model = model
        self.epochs = epochs
        self.data = dataset.DataLoader(data_path)
        self.training_data = \
            utils.data.DataLoader(self.data, batch_size = 32, shuffle = True)
        
        self.save = save
        
        # reload the pixel CNN model if the save file specified exists
        # else make a new one
        if save != None and os.path.isfile(save) == True:
            cnn = modules.PixelCNN(num_embeddings)
            state_dict = torch.load(save, map_location="cpu")
            cnn.load_state_dict(state_dict)
            cnn.to(self.device)
            cnn.eval()
            self.PixelCNN_model = cnn
        else:
            self.PixelCNN_model = \
                modules.PixelCNN(self.num_embeddings).to(self.device)
        
        self.optimizer = \
            torch.optim.Adam(self.PixelCNN_model.parameters(), lr = lr)
    """
    train the pixelCNN model.
    
    """
    def train(self):
        epoch = 0
        training_loss_arr = []

        while epoch != self.epochs:
            
            sub_step = 0
            for i, _ in self.training_data:
                i = i.view(-1, 3, 128, 128)
                
                with torch.no_grad():
                    # get encoder and get the image at the point when it is in 
                    # its latent form
                    encoder = self.model.get_encoder().to(self.device)
                    VQ = self.model.get_VQ().to(self.device)
                    i = i.to(self.device)
                    encoded = encoder(i)
                    encoded = encoded.permute(0, 2, 3, 1).contiguous()
                    flat_encoded  = encoded.reshape(-1, VQ.embedding_dim)
                    # get the codebook indices here
                    a, b = VQ.argmin_indices(flat_encoded)
                    b = b.view(-1, 32, 32)
                    c = nn.functional.one_hot(b, num_classes = 
                                              self.num_embeddings).float()
                    c = c.permute(0, 3, 1, 2)
                # train the pixel CNN with the codebook indices
                prior = self.PixelCNN_model(c)
                self.optimizer.zero_grad()
                # total loss using cross entropy loss
                total_loss = nn.functional.cross_entropy(prior, b)
                # update loss and optimizer
                total_loss.backward()
                self.optimizer.step()
                
                if sub_step == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    # append loss
                    training_loss_arr.append(total_loss.item())
                    # save if there is a file path specified
                    if self.save != None:
                        torch.save(self.PixelCNN_model.state_dict(), self.save)
                    # generate an image with the current state of the model
                    predict.gen_image(self.model_path, self.save, 
                                      self.num_embeddings, self.latent_dim)
                    

                sub_step += 1
                
                
            epoch += 1         
                    
        # save anyways if loop finishes by itself
        if self.save != None:
            
            torch.save(self.PixelCNN_model.state_dict(), self.save)
        # plot the training loss over number of epochs
        plt.plot(np.arange(0, self.epochs), training_loss_arr)
        plt.title("Training Loss vs Number of Epochs")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Training Loss")
        plt.show()
        




 
