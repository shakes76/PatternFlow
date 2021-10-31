import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, 
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1,
                      bias=False)
        )

    def forward(self, x):
        return x + self.block(x)

class VQVAE(nn.Module):
    def __init__(self, img_channels, latent_size, latent_dim):
        super(VQVAE, self).__init__()
        
        self.K = latent_size
        self.D = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, self.D//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.D//2, self.D, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(self.D, self.D), 
            nn.ReLU(),
            ResidualBlock(self.D, self.D), 
            nn.ReLU(),
        )
        
        self.codebook = nn.Embedding(self.K, self.D)
        self.codebook.weight.data.uniform_(-1/self.K, 1/self.K)
        
        self.decoder = nn.Sequential(
            ResidualBlock(self.D, self.D), 
            nn.ReLU(),
            ResidualBlock(self.D, self.D), 
            nn.ReLU(),
            nn.ConvTranspose2d(self.D, self.D//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.D//2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU() 
        )
        
    def vector_quantize(self, z_e):
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_shape = z_e.shape

        flat_z_e = z_e.view(-1, self.D)
        
        distances = (torch.sum(flat_z_e**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z_e, self.codebook.weight.t()))
        
        # q = torch.argmin(distances, dim=1, keepdim=True)
        # q_ont_hot = torch.zeros(distances.shape)
        # q_ont_hot.scatter_(1, q, 1)
        
        # z_q = torch.matmul(q_ont_hot, self.codebook.weight).view(z_e_shape)
        
        z_q = self.codebook(q)
        
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commit_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + commit_loss
        
        z_q = z_e + (z_q - z_e).detach()
        
        return q, vq_loss, z_q.permute(0, 3, 1, 2).contiguous()
    
    def forward(self, imgs):
        z_e = self.encoder(imgs)
        _, vq_loss, encoded = self.vector_quantize(z_e)
        decoded = self.decoder(encoded)
        
        return encoded, decoded, vq_loss
    
    