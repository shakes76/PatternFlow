import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
import numpy as np
import os
import modules
import dataset
import visualise as vis
import predict

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


NUM_EMBEDDINGS = 64
LATENT_DIM = 16
        
class VQ_Training():
    
    
    def __init__(self, learning_rate, epochs, train_path, test_path, num_embeddings,
                 latent_dim, save = None, visualise = False):
        super(VQ_Training).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.data = dataset.DataLoader(train_path)
        self.data2 = dataset.DataLoader(test_path)
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        
        self.training_data = \
            utils.data.DataLoader(self.data, batch_size = 16, shuffle = True)
        
        self.testing_data = \
            utils.data.DataLoader(self.data, batch_size = 1, shuffle = True)
        
        self.model = modules.VQVAE(self.num_embeddings,self.latent_dim,0.25).to(self.device)
        
        self.optimizer = \
            torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        
        self.save = save
        self.visualise = visualise
        if self.visualise == True:
            self.visualiser = vis.VQVAE_Visualise(self.model, self.data)
            
    def train(self):
        epoch = 0
        loss = []
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
                        f"Epoch [{epoch}/{self.epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    loss.append(total_loss.item())
                    
                    if self.visualise == True:
                        self.visualiser.VQVAE_discrete((0,0))
                        self.visualiser.visualise_VQVAE((0,0))
        
                
                
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
    
    If Visualise is selected, then it visulises the decoded outputs, 
    
    """
    def test(self):
        ssim_list = []
        max_ssim = 0
        for i, _ in self.testing_data:
            i = i.view(-1, 3, 256, 256).to(self.device).detach()
            real_grid = torchvision.utils.make_grid(i, normalize = True)
            with torch.no_grad():
                decoded_img , _ = self.model(i)
                decoded_img = decoded_img.view(-1, 3, 256,256).to(self.device).detach()
                decoded_grid = \
                    torchvision.utils.make_grid(decoded_img, normalize = True)
                decoded_grid = decoded_grid.to("cpu").permute(1,2,0)
                real_grid = real_grid.to("cpu").permute(1,2,0)
                val =\
                    ssim(real_grid.numpy(), decoded_grid.numpy(), channel_axis = -1)
                if max_ssim < val:
                    max_ssim = val
                ssim_list.append(val)
        print("The Average SSIM for the test data is " + (str) (np.average(ssim_list)))
        print("The maximum SSIM is " + (str) (np.amax(ssim_list)))
        
        
data_path = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\keras_png_slices_data\train"
data_path2 = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\keras_png_slices_data\test"

trained = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\trained_model\bruh.pt"
lr = 0.001
epochs  = 15

#trainer = VQ_Training(lr, epochs, data_path, data_path2, NUM_EMBEDDINGS, LATENT_DIM,
  #                    save = trained, visualise=True)
#trainer.train()
#trainer.test()
class PixelCNN_Training():
    
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
            
    def train(self):
        epoch = 0
        training_loss_arr = []

        while epoch != self.epochs:
            
            sub_step = 0
            for i, _ in self.training_data:
                i = i.view(-1, 3, 256, 256)
                
                with torch.no_grad():
                    encoder = self.model.get_encoder().to(self.device)
                    VQ = self.model.get_VQ().to(self.device)
                    i = i.to(self.device)
                    encoded = encoder(i)
                    encoded = encoded.permute(0, 2, 3, 1).contiguous()
                    flat_encoded  = encoded.reshape(-1, VQ.embedding_dim)
                    
                    a, b = VQ.argmin_indices(flat_encoded)
                    b = b.view(-1, 64, 64)
                    c = nn.functional.one_hot(b, num_classes = self.num_embeddings).float()
                    c = c.permute(0, 3, 1, 2)
                    #b = b.permute(1, 0, 2, 3).contiguous()
                prior = self.PixelCNN_model(c)
                self.optimizer.zero_grad()
       
                total_loss = nn.functional.cross_entropy(prior, b)
                
                total_loss.backward()
                self.optimizer.step()
                
                if sub_step == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    
                    training_loss_arr.append(total_loss.item())
                    if self.save != None:
                        torch.save(self.PixelCNN_model.state_dict(), self.save)
                    predict.gen_image(self.model_path, self.save, self.num_embeddings, 
                              self.latent_dim)
                    

                sub_step += 1
                
                
            epoch += 1
        #save if user interrupts
         
                    
        # save anyways if loop finishes by itself
        if self.save != None:
            
            torch.save(self.PixelCNN_model.state_dict(), self.save)

        plt.plot(np.arange(0, self.epochs), training_loss_arr)
        plt.title("Training Loss vs Number of Epochs")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Training Loss")
        plt.show()
        

        
save_model =  r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\trained_model\cnn_model3_5.pt"
pixel_cnn_trainer = PixelCNN_Training(0.0005, 500, 
                                      trained,data_path, num_embeddings = 64, 
                                      latent_dim = 16, save = save_model) 
                                      
pixel_cnn_trainer.train()



 
