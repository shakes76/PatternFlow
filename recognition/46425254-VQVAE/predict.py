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

"""
Given a VQVAE model and a pixel CNN model, generate a brain image.

Parameters:
    train_path -> the path to the trained VQVAE model
    model_path -> the path to the trained PixelCNN model
    num_embeddings -> the number of embeddings of the VQVAE
    latent_dim -> the latent space of the VQVAE
"""

def gen_image(train_path, model_path, num_embeddings, latent_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = modules.VQVAE(num_embeddings,latent_dim,0.25)
    state_dict = torch.load(train_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    cnn = modules.PixelCNN(num_embeddings)
    state_dict = torch.load(model_path, map_location="cpu")
    cnn.load_state_dict(state_dict)
    cnn.to(device)
    cnn.eval()
 
    prior = torch.zeros((1, num_embeddings, 64, 64), device = device)
    
    _, channels, rows, cols = prior.shape
    
    
    with torch.no_grad():
        for i in range(rows):
            for j in range(cols):
                out = cnn(prior.float())
                pixel = out[:, :, i, j]
                values = torch.unique(pixel)
                values = torch.sort(values, 0).values
                #print(values.shape)
                value = values[58]
                pixel[pixel < value] = -999999
                distribution = \
                     torch.distributions.categorical.Categorical(logits = pixel)
                
                sampled = distribution.sample()
                sampled = nn.functional.one_hot(sampled, \
                                                num_classes = num_embeddings)
                prior[:, :, i, j] = sampled
             

    _, ax = plt.subplots(1,2)
    ax[0].imshow(prior.argmax(1).view(64,64).to("cpu"))
    ax[0].title.set_text("Latent Generation")

    prior = prior.view(1,num_embeddings,-1)
    prior = prior.permute(0,2,1).float()
    quantized_bhwc = torch.matmul(prior, 
                                  model.get_VQ().embedding_table.weight)
    
    quantized_bhwc = quantized_bhwc.view(1, 64, 64, latent_dim)
    quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
    
    decoded = model.get_decoder()(quantized).to(device)
    
    decoded = decoded.view(-1, 3, 256,256).to(device).detach()
    
    decoded_grid = \
        torchvision.utils.make_grid(decoded, normalize = True)
    decoded_grid = decoded_grid.to("cpu").permute(1,2,0)
    
    ax[1].imshow(decoded_grid)
    ax[1].title.set_text("Decoded Generation")
    plt.show()
    
    
"""
Given a VQVAE model and some test data, test SSIM of decoded OASIS images

Parameters:
    VQVAE_path -> the path to the trained model in .pt format
    data -> the path of the dataset of OASIS images
    
    coords -> a tuple containing whichever image you want to run through the
    VQVAE
    
"""

def VQVAE_predict(VQVAE_path, num_embeddings, latent_dim, data, coords):
    ds = dataset.DataLoader(data_path=data)
    visualiser = \
        vis.VQVAE_Visualise(VQVAE_path, num_embeddings, latent_dim, ds)
    
    visualiser.VQVAE_discrete(coords)
    visualiser.visualise_VQVAE(coords)

CNN =  r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\trained_model\cnn_model2.pt"
train = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\keras_png_slices_data\train"
test = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\keras_png_slices_data\test"

VQVAE = r"C:\Users\blobf\COMP3710\PatternFlow\recognition\46425254-VQVAE\trained_model\bruh.pt"

#VQVAE_predict(VQVAE, 64, 16, test, (0,0))
