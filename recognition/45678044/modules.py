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
    
class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)
    
class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual


        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h):
        if self.mask_type == 'A':
            self.make_causal()

        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert)

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz)
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h
    
class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # self.norm = nn.BatchNorm2d(dim)
        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        # self.apply(weights_init)

    def forward(self, x):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        # x = self.norm(x)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h)

        return self.output_conv(x_h)

    def generate(self, shape=(64, 64), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x