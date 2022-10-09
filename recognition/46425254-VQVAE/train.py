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
        
        while epoch != self.epochs:
            
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
epochs  = 15

#trainer = VQ_Training(lr, epochs, path, save = trained, visualise=True)
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
        self.epochs = epochs
        data = dataset.DataLoader(data_path)
        self.training_data = \
            utils.data.DataLoader(data, batch_size = 16, shuffle = True)
        
        
        
        
       # self.loss = nn.functional.cross_entropy()
        self.save = save
        
        self.PixelCNN_model = modules.PixelCNN().to(self.device)
        self.optimizer = \
            torch.optim.Adam(self.PixelCNN_model.parameters(), lr = 0.0001)
            
    def train(self):
        epoch = 0
        
        while epoch != self.epochs:
            
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
                   # print(a.shape)
                    
                    a = a.view(-1, 128, 64,64).float()
                    #b = torch.stack((b,b,b),0)
                    #print(a.shape)
                    #b = b.permute(1, 0, 2, 3).contiguous()
                cnn_outputs = self.PixelCNN_model(a)
                
                #print(b.shape)
                #reset the optimizer gradients to 0 to avoid resuing prev iteration's 
                
                
                
                #calculate reconstruction loss
                
                
                #calculate total loss
                self.optimizer.zero_grad()
                total_loss = nn.functional.cross_entropy(cnn_outputs, a)
                
                #update the gradient 
                total_loss.backward()
                self.optimizer.step()
                
                if sub_step == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    
                   # if self.visualise == True:
                   #     self.visualiser.VQVAE_discrete((0,0))
        
                
                
                sub_step += 1
            epoch += 1
        #save if user interrupts
         
                    
        # save anyways if loop finishes by itself
        if self.save != None:
            
            torch.save(self.PixelCNN_model.state_dict(), self.save)
        
def gen_image(train_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = modules.VQVAE()
    state_dict = torch.load(train_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    
    cnn = modules.PixelCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    cnn.load_state_dict(state_dict)
    cnn.to(device)
    
    prior = torch.zeros((1, 128, 64, 64), device = device)
    
    _, _, rows, cols = prior.shape
    for i in range(rows):
        for j in range(cols):
            torch.cuda.empty_cache()
            probs = cnn(prior)

            prior[:, :, i , j] = probs[:, :, i, j]
           
            
    quantized_bhwc = torch.matmul(prior, 
                                  model.embedding_table.weight).view(1, 64, 64, 64)
    
    quantized = quantized_bhwc.permute(0, 3, 1, 2).continguous()
    
    decoded = model.get_decoder(quantized).to(device)
    
    decoded = decoded.view(-1, 3, 256,256).to(device).detach()
    
    decoded_grid = \
        torchvision.utils.make_grid(decoded, normalize = True)
    decoded_grid = decoded_grid.to("cpu").permute(1,2,0)
    
    plt.imshow(decoded_grid)
    plt.show()
        
save_model =  r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\trained_model\bruh2.pt"
pixel_cnn_trainer = PixelCNN_Training(lr, 10, trained, path, save = save_model)
pixel_cnn_trainer.train()
#gen_image(trained, save_model)


 
