import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    """ Weight initialiser function """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


class Embedding(nn.Module):
    """ 
    Embedding class to discretise the distance between continuous embeddings
    and the encoded outputs. 

    Parameters:
        K: the size of the latent space.
        D: the dimension of each embedding.
        commitement_cost: commitment cost (beta) used in loss term.
    """
    def __init__(self, K, D, commitment_cost=1.0):
        super().__init__()
        self.K = K
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        # Reshape inputs from BCHW to BHWC
        x = x.permute(0, 2, 3, 1).contiguous()

        flattened = x.view(-1, self.embedding.weight.size(1))

        # Calculate distances to find the closest codebook vectors
        distances = (
            torch.sum(flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flattened, self.embedding.weight.t())
        )

        # We get the indices from argmin
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.K, device=DEVICE
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantise the vector by multiplying the encodings with the embeddings
        quantised = torch.matmul(encodings, self.embedding.weight).view(x.shape)

        # Calculate the loss. Note that the paper has log likelihood as the
        # loss but minimising the mean squared error is equivalent.
        commit_loss = F.mse_loss(quantised.detach(), x)
        vq_loss = F.mse_loss(quantised, x.detach())

        quantised = x + (quantised - x).detach()

        return (
            vq_loss + self.commitment_cost * commit_loss,
            quantised.permute(0, 3, 1, 2).contiguous(),
            encodings,
            torch.argmin(distances, dim=1),
        )

    def quantise(self, x, batch_size):
        """ Returns the embedding from given codebook indices. """
        encoding_indices = x.unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.K, device=DEVICE
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantised = torch.matmul(encodings, self.embedding.weight).view(
            batch_size, 64, 64, 256
        )
        return quantised.permute(0, 3, 1, 2).contiguous()


class ResBlock(nn.Module):
    """ The ResNet block """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    """ 
    The encoder network that takes an image and maps it to the latent space.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 4, 2, 1),
            
            ResBlock(out_channels),
            ResBlock(out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """ 
    The decoder network that takes a latent space sample and maps it back to 
    the original space. 
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(in_channels),
            ResBlock(in_channels),

            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    """ The VQ-VAE Network """
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.codebook = Embedding(K, out_channels)
        self.decoder = Decoder(out_channels, in_channels)

        self.apply(weights_init)

    def forward(self, x):
        loss, quantised, _, _ = self.codebook(self.encoder(x))
        x_recon = self.decoder(quantised)
        return loss, x_recon


class Discriminator(nn.Module):
    """ 
    The discriminator network that take an image and returns the probability 
    that the image is real. 
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.net(input)


class Generator(nn.Module):
    """ 
    The generator network that generates an image from the given latent space. 
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.net(input)
