import numpy as np
import torch
import torch.nn.functional as F
from dataset import create_train_test_loaders
from modules import VQVAE


def train_VQVAE():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_train_test_loaders()
    LEARNING_RATE = 1e-3
    NUM_EMBEDDINGS = 512  # number of embeddings for codebook
    EMBEDDING_DIM = 64  # embedding dimension
    BETA_COST = 0.25  # commitment cost for VQ
    DATA_VARIANCE = 0.0338  # evaluated seperately on training data
    NUM_EPOCHS = 1500  # number of epochs to train the VQVAE
    train_res_recon_error = []

    model = VQVAE(NUM_EMBEDDINGS, EMBEDDING_DIM, BETA_COST).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train() # sets the model in training model so gradients are tracked

    for i in range(NUM_EPOCHS):
        data = next(iter(train_loader)) # gets the next training data
        data = data.to(device)
        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / DATA_VARIANCE # reconstructed loss with is an MSE loss scaled by var
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        train_res_recon_error.append(recon_error.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print()

