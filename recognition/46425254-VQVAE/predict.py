import torch
import torch.nn as nn
import torchvision

import modules
import dataset
import visualise as vis

import matplotlib.pyplot as plt

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
    # load the VQVAE model
    model = modules.VQVAE(num_embeddings,latent_dim,0.25)
    state_dict = torch.load(train_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # load the PixelCNN model
    cnn = modules.PixelCNN(num_embeddings)
    state_dict = torch.load(model_path, map_location="cpu")
    cnn.load_state_dict(state_dict)
    cnn.to(device)
    cnn.eval()
    # start with a zeros tensor
    prior = torch.zeros((1, num_embeddings, 32, 32), device = device)
    
    _, channels, rows, cols = prior.shape
    
    # get probable pixels for each row, col using categorical distribution and
    # pixelCNN mask layers
    with torch.no_grad():
        for i in range(rows):
            for j in range(cols):
                out = cnn(prior.float())

                out = out.permute(0,2,3,1).contiguous()
                distribution = \
                    torch.distributions.categorical.Categorical(logits = out)
                
                sampled = distribution.sample()
                sampled = \
                    nn.functional.one_hot(sampled, num_classes \
                                          = num_embeddings
                                          ).permute(0, 3, 1, 2).contiguous()
                # write each row col with sampled row col
                prior[:, :, i , j] = sampled[:, :, i, j]

             

    _, ax = plt.subplots(1,2)
    # plot latent generation
    ax[0].imshow(prior.argmax(1).view(32,32).to("cpu"))
    ax[0].title.set_text("Latent Generation")
    # process the latent generation so it can be decoded, just like in VQVAE
    prior = prior.view(1,num_embeddings,-1)
    prior = prior.permute(0,2,1).float()
    quantized_bhwc = torch.matmul(prior, 
                                  model.get_VQ().embedding_table.weight)
    # rearranging and converting dimensions, same as the VQVAE forward process
    quantized_bhwc = quantized_bhwc.view(1, 32, 32, latent_dim)
    quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
    
    decoded = model.get_decoder()(quantized).to(device)
    
    decoded = decoded.view(-1, 3, 128,128).to(device).detach()
    
    decoded_grid = \
        torchvision.utils.make_grid(decoded, normalize = True)
    decoded_grid = decoded_grid.to("cpu").permute(1,2,0)
    # plot the decoded geneartion
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


