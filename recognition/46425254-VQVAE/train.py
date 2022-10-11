import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
import numpy as np

import modules
import dataset
import visualise as vis

import CNNcopy
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

save_model =  r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\trained_model\bruh2.pt"

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
        self.data = dataset.DataLoader(data_path)
        self.training_data = \
            utils.data.DataLoader(self.data, batch_size = 16, shuffle = True)
        
        
        
        
       # self.loss = nn.functional.cross_entropy()
        self.save = save
        
        self.PixelCNN_model = modules.PixelCNN().to(self.device)
        self.optimizer = \
            torch.optim.Adam(self.PixelCNN_model.parameters(), lr = lr)
            
    def train(self):
        epoch = 0
        training_loss_arr = []
        loss = 0.0
        total_epochs = np.linspace(0, self.epochs, self.epochs)
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
                    #print(b.shape)
                    #a = a.view(-1, 128, 64,64).float()
                    #a = a.argmax(1)
                    #b = torch.stack((b,b,b),0)
                    #print(a.shape)
                    b = b.view(-1, 64, 64)
                    c = nn.functional.one_hot(b, num_classes = 128).float()
                    c = c.permute(0, 3, 1, 2)
                    #b = b.permute(1, 0, 2, 3).contiguous()
                cnn_outputs = self.PixelCNN_model(c)
                #plt.imshow(cnn_outputs.argmax(1)[0].cpu().detach().numpy())
                
                #print(b.shape)
                #reset the optimizer gradients to 0 to avoid resuing prev iteration's 
                #plt.show()               
                #calculate reconstruction loss
                #calculate total loss
                self.optimizer.zero_grad()
                total_loss = nn.functional.cross_entropy(cnn_outputs, b)
                
                total_loss.backward()
                self.optimizer.step()
                
                if sub_step == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    
                    training_loss_arr.append(total_loss.item())
                  #  if self.save != None:
                        
                 #       torch.save(self.PixelCNN_model.state_dict(), self.save)
                    
                   # gen_image(trained, save_model)
                    
                    #if self.visualise == True:
                     #   self.visualiser.VQVAE_discrete((0,0))
                sub_step += 1
                
                
            epoch += 1
        #save if user interrupts
         
                    
        # save anyways if loop finishes by itself
        if self.save != None:
            
            torch.save(self.PixelCNN_model.state_dict(), self.save)

        plt.plot(np.arange(0, self.epochs12), training_loss_arr)
        plt.show()
        
def gen_image(train_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = modules.VQVAE()
    state_dict = torch.load(train_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    cnn = modules.PixelCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    cnn.load_state_dict(state_dict)
    cnn.to(device)
    cnn.eval()
    
    prior = torch.zeros((1, 128, 64, 64), device = device)
    
    _, channels, rows, cols = prior.shape
    with torch.no_grad():
        for i in range(rows):
            for j in range(cols):
                # argmax removes things that is not predicted
                out = cnn(prior).argmax(1)
                
                #argmax(1)
                #convert it back into 1 hot format
                #print(torch.multinomial(probs, 1))
                probs = nn.functional.one_hot(out, num_classes = 128
                                              ).permute(0, 3, 1, 2).contiguous()
                
                prior[:, :, i , j] = probs[:, :, i, j]
                
            
    
    #prior = prior.argmax(1)
    #plt.imshow(prior[0].to("cpu"))
    #plt.show()
    #prior = nn.functional.one_hot(prior, num_classes= 128)
    #prior = prior.permute(0, 3, 1, 2)
    prior = prior.view(1,128,-1)
    
    prior = prior.permute(0,2,1).float()
    quantized_bhwc = torch.matmul(prior, 
                                  model.get_VQ().embedding_table.weight)
    quantized_bhwc = quantized_bhwc.view(1, 64, 64 ,16)
    quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
    
    decoded = model.get_decoder()(quantized).to(device)
    
    decoded = decoded.view(-1, 3, 256,256).to(device).detach()
    
    decoded_grid = \
        torchvision.utils.make_grid(decoded, normalize = True)
    #print(decoded_grid.shape)
    decoded_grid = decoded_grid.to("cpu").permute(1,2,0)
    
    plt.imshow(decoded_grid)
    plt.show()
        
save_model =  r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\trained_model\bruh2.pt"
pixel_cnn_trainer = PixelCNN_Training(0.0005, 4, trained, path, save = save_model)
pixel_cnn_trainer.train()
#gen_image(trained, save_model)


 
