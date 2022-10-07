import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
import numpy as np

import modules
import dataset

import matplotlib.pyplot as plt



        
class Training():
    
    
    def __init__(self, learning_rate, epochs, path, save = None):
        super(Training).__init__()
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

        
    def train(self):
        fixed_noise = torch.randn((1, 256, 32, 32)).to(self.device)
        model = self.model
        adam = self.optimizer
        for epoch in range(epochs):
            sub_step = 0
            for i, _ in self.training_data:
                i = i.view(-1, 3, 256, 256).to(self.device)
                decoder_outputs, VQ_loss = model(i)
                #reset the optimizer gradients to 0 to avoid resuing prev iteration's 
                adam.zero_grad()
                
                #calculate reconstruction loss
                recon_loss = nn.functional.mse_loss(decoder_outputs, i)
                
                #calculate total loss
                total_loss = recon_loss + VQ_loss
                
                #update the gradient 
                total_loss.backward()
                adam.step()
                
                if sub_step == 0:
                    print(
                        f"Epoch [{epoch}/{epochs}] \ " 
                        f"Loss : {total_loss:.4f}"
                    )
                    
                    
                    # generate an image every epoch
                    real_img = self.data[0][0]
                    real_img = real_img.to("cpu")
                    _, ax = plt.subplots(1,2)
                    real_grid = torchvision.utils.make_grid(real_img, normalize = True)
                    ax[0].imshow(real_grid.permute(1,2,0))
                    real_img = real_img.view(-1, 3, 256,256).to(self.device)
                    decoded_img , _ = model(real_img)
                    decoded_grid = torchvision.utils.make_grid(decoded_img.to("cpu"), normalize = True)
                    ax[1].imshow(decoded_grid.permute(1,2,0))
                    plt.show()
                    
                    
                    
                sub_step += 1
        
        #save the model if a path is specified for save
        if self.save != None:
            torch.save(model.state_dict(), self.save)
            
    
    def gen_fake(self, fixed_noise):
        
        fake = self.model.decoder(fixed_noise).reshape(-1, 3, 256, 256)
        
        fake_img_grid = torchvision.utils.make_grid(fake, normalize = True)
        fake_img_grid = fake_img_grid.cpu().numpy()
        plt.imshow(np.transpose(fake_img_grid, (1,2,0)), interpolation='nearest')
        plt.show()
        
        
        
        
        
        
        
path = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\keras_png_slices_data\train"
lr = 0.0002
epochs  = 50

trainer = Training(lr, epochs, path)
trainer.train()

trainer.gen_fake(1)
