from imports import *

"""
Defines a (linear) beta schedule

Implemented as per the original paper,
also as described in the author's blog post:
https://huggingface.co/blog/annotated-diffusion
"""
def beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end   = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Define useful CONSTANTS
TIMESTEPS = 1000
BETAS = beta_schedule(TIMESTEPS)
ALPHAS = 1. - BETAS
ALPHAS_CUML_PRODUCT = torch.cumprod(ALPHAS, axis=0)
ALPHAS_CUML_PRODUCT_PREV = F.pad(ALPHAS_CUML_PRODUCT[:-1], (1, 0), value=1.)
SQRT_RECIP_ALPHAS = torch.sqrt(1. / ALPHAS)
SQRT_ALPHAS_CUML_PRODUCT = torch.sqrt(ALPHAS_CUML_PRODUCT)
SQRT_ONE_MINUS_ALPHAS_CUML_PRODUCT = torch.sqrt(1. - ALPHAS_CUML_PRODUCT)
POSTERIOR_VAR = BETAS * (1. - ALPHAS_CUML_PRODUCT_PREV) / (1. - ALPHAS_CUML_PRODUCT)

"""
Extract the appropriate t index for a batch of indices

Taken from original paper, as provided in the author's blog post:
https://huggingface.co/blog/annotated-diffusion
"""
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

"""
Simulates forward diffusion

Taken from the original paper, also as described in the author's blog post:
https://huggingface.co/blog/annotated-diffusion
"""
def forward_diffusion(x_0, t, device="cuda", noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = extract(SQRT_ALPHAS_CUML_PRODUCT, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(SQRT_ONE_MINUS_ALPHAS_CUML_PRODUCT, t, x_0.shape)

    return sqrt_alphas_cumprod_t * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t * noise.to(device), noise

"""
Sinusoidal position embeddings

Taken from original paper,
also as described in the author's blog post:
https://huggingface.co/blog/annotated-diffusion
"""
class PositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        x = math.log(10000) / (half_dim - 1)
        x = torch.exp(torch.arange(half_dim, device=device) * -x)
        x = time[:, None] * x[None, :]
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        return x

"""
Building block of the UNet Model
"""
class Block(nn.Module):
    def __init__(self, in_c, out_c, up=False, time_emb=32):
        super().__init__()
        # define layers
        self.time_mlp =  nn.Linear(time_emb, out_c)
        self.conv1  = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_c)
        self.bnorm2 = nn.BatchNorm2d(out_c)
        self.relu   = nn.ReLU()

    def forward(self, x, t):
        # conv x
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        # time embedding
        time_emb = self.time_mlp(t)
        time_emb = self.relu(time_emb)
        # extend dimensions
        time_emb = time_emb[(..., ) + (None,) * 2]
        # add time channel
        x = x + time_emb
        # final conv
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu(x)
        return x

"""
Building block of the encoder network
"""
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, up=False, time_emb=32):
        super().__init__()
        # define layers
        self.conv = Block(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv(x, t)
        skip = x.clone()
        x = self.pool(x)
        return x, skip

"""
Building block of the decoder network
"""
class DecoderBlock(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.up    = nn.ConvTranspose2d(num_in, num_out, kernel_size=2, stride=2, padding=0)
        self.block = Block(num_out + num_out, num_out)

    def forward(self, x, t, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.block.forward(x, t)
        return x

"""
The final UNet network
"""
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # useful information
        down_channels = (1, 64, 128, 256, 512)
        up_channels =   (1024, 512, 256, 128, 64)

        # time embedding
        time_dim = 32
        self.time_mlp = nn.Sequential(
            PositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )

        self.downs = nn.ModuleList([EncoderBlock(down_channels[i], down_channels[i+1]) for i in range(len(down_channels)-1)])
        self.ups   = nn.ModuleList([DecoderBlock(up_channels[i], up_channels[i+1]) for i in range(len(up_channels)-1)])
        
        self.bottle_neck = Block(down_channels[-1], up_channels[0])

        self.out = nn.Conv2d(up_channels[-1], 1, kernel_size=1, padding=0)

    def forward(self, x, t):
        t = self.time_mlp(t)

        # Encoder
        residuals = []
        for down in self.downs:
            x, skip = down(x, t)
            residuals.append(skip)

        # Bottle neck
        x = self.bottle_neck(x, t)

        # Decoder
        for up in self.ups:
            x = up(x,t,residuals.pop())

        return self.out(x)

"""
Loss function
Used to optimize the UNet network

Employs F1 loss, between the predicted and true noise
"""
def get_loss(model, x_0, t):
    x_batch_sample, noise = forward_diffusion(x_0, t)
    noise_prediction = model(x_batch_sample, t)
    return F.l1_loss(noise, noise_prediction)
