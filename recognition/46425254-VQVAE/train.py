import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
import numpy as np

import modules
import dataset
import visualise as vis


import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim



        
class VQ_Training():
    
    
    def __init__(self, learning_rate, epochs, path, save = None, visualise = False):
        super(VQ_Training).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.data = dataset.DataLoader(path)
        self.training_data = \
            utils.data.DataLoader(self.data, batch_size = 15, shuffle = True)
            
        self.model = modules.VQVAE().to(self.device)
        
        self.optimizer = \
            torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        
        self.save = save
        self.visualise = visualise
        if self.visualise == True:
            self.visualiser = vis.Visualise(self.model, self.data)
            
    def train(self):
        epoch = 0
        
        while epoch != epochs:
            
            sub_step = 0
            for i, _ in self.training_data:
                i = i.view(-1, 3, 256, 256).to(self.device)
                
                decoder_outputs, VQ_loss = self.model(i)
                #reset the optimizer gradients to 0 to avoid resuing prev iteration's 
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
                        f"Epoch [{epoch}/{epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    
                    if self.visualise == True:
                        self.visualiser.VQVAE_discrete((0,0))
        
                
                
                sub_step += 1
            epoch += 1
        #save if user interrupts
         
                    
        # save anyways if loop finishes by itself
        if self.save != None:
            
            torch.save(self.model.state_dict(), self.save)
            
    

        
        
        
        
        
path = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\keras_png_slices_data\train"

trained = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\trained_model\bruh.pt"
lr = 0.0002
epochs  = 25

#trainer = VQ_Training(lr, epochs, path, save = trained, visualise=False)
#trainer.train()


class PixelCNN_Training():
    
    def __init__(self, lr, epochs, model_path, data_path, save = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.epochs = epochs
        model = modules.VQVAE()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.model = model
        
        data = dataset.DataLoader(data_path)
        self.training_data = \
            utils.data.DataLoader(data, batch_size = 16, shuffle = True)
        
        
        
        
        self.loss = nn.CrossEntropyLoss()
        self.save = save
        
        self.PixelCNN_model = modules.PixelCNN().to(self.device)
        self.optimizer = \
            torch.optim.Adam(self.PixelCNN_model.parameters(), lr = 0.0001)
            
    def train(self):
        epoch = 0
        
        while epoch != epochs:
            
            sub_step = 0
            for i, _ in self.training_data:
                i = i.view(-1, 3, 256, 256)
                
                encoder = self.model.get_encoder().to(self.device)
                VQ = self.model.get_VQ().to(self.device)
                with torch.no_grad():
                    i = i.to(self.device)
                    encoded = encoder(i)
                    encoded = encoded.permute(0, 2, 3, 1).contiguous()
                    flat_encoded  = encoded.reshape(-1, VQ.embedding_dim)
                    a, b = VQ.argmin_indices(flat_encoded)
                    
                    b = b.view(-1, 64,64).float()
                    b = torch.stack((b,b,b),0)
                    b = b.permute(1, 0, 2, 3).contiguous()
                cnn_outputs = self.PixelCNN_model(b)
                #reset the optimizer gradients to 0 to avoid resuing prev iteration's 
                
                
                
                #calculate reconstruction loss
                
                
                #calculate total loss
                
                total_loss = self.loss(cnn_outputs, b)
                self.optimizer.zero_grad()
                #update the gradient 
                total_loss.backward()
                self.optimizer.step()
                
                if sub_step == 0:
                    print(
                        f"Epoch [{epoch}/{epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    
                   # if self.visualise == True:
                   #     self.visualiser.VQVAE_discrete((0,0))
        
                
                
                sub_step += 1
            epoch += 1
        #save if user interrupts
         
                    
        # save anyways if loop finishes by itself
        if self.save != None:
            
            torch.save(self.model.state_dict(), self.save)
        
pixel_cnn_trainer = PixelCNN_Training(lr, epochs, trained, path)
pixel_cnn_trainer.train()


 
