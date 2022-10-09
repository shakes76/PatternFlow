from imports import *
from dataset import ImageLoader

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
Load in image(s) and transform data via custom ImageLoader, and create our DataLoader
"""
def load_dataset(batch_size=8, image_resize=256, ad_train_path="ADNI_DATA/AD_NC/train/AD", nc_train_path="ADNI_DATA/AD_NC/train/NC"):
    # transform image from [1,255] to [0,1], and scale linearly into [-1,1]
    transform = Compose([
                    ToPILImage(),
                    Grayscale(),  # ensure images only have one channel
                    Resize(image_resize),  # ensure all images have same size
                    CenterCrop(image_resize),
                    ToTensor(),
                    Lambda(lambda t: (t * 2) - 1),  # scale linearly into [-1,1]
                ])
    # Load data, with the above transform applied
    train_ad_imgs = ImageLoader(ad_train_path,
                                transform=transform)
    train_nc_imgs = ImageLoader(nc_train_path,
                                transform=transform)
    # combine ad and nc train datasets, to increase total number of images for training
    total_imgs = torch.utils.data.ConcatDataset([train_ad_imgs, train_nc_imgs])
    return DataLoader(total_imgs, batch_size=batch_size, shuffle=False, num_workers=1)

"""
Sinusoidal position embeddings

Taken from original paper,
also as described in the author's blog post:
https://huggingface.co/blog/annotated-diffusion
"""
class SinusoidalPositionEmbeddings(nn.Module):
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
    def __init__(self, in_c, out_c, time_emb, up=False):
        super().__init__()
        # define layers
        self.time_mlp =  nn.Linear(time_emb, out_c)
        if up:
            self.conv1     = nn.Conv2d(2*in_c, out_c, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(out_c, out_c, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1     = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_c, out_c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_c)
        self.bnorm2 = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # conv x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bnorm1(x)
        # time embedding
        time_emb = self.time_mlp(t)
        time_emb = self.relu(time_emb)
        # extend dimensions
        time_emb = time_emb[(..., ) + (None,) * 2]
        # add time channel
        x = x + time_emb
        # final conv
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bnorm2(x)
        # return up/down batch_sample
        return self.transform(x)

"""
The contractive path of the UNet
"""
class Encoder(nn.Module):
    def __init__(self, time_dim=32, down_channels=(64, 128, 256, 512, 1024)) -> None:
        super().__init__()
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_dim) for i in range(len(down_channels)-1)])

    def forward(self, x, t):
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        return x, residuals

"""
The expansive path of the UNet
"""
class Decoder(nn.Module):
    def __init__(self, time_dim=32, up_channels=(1024, 512, 256, 128, 64)) -> None:
        super().__init__()
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_dim, up=True) for i in range(len(up_channels)-1)])

    def forward(self, x, t, residuals):
        for up in self.ups:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = up(x,t)
        return x

"""
The UNet model
"""
class UNet(nn.Module):
    def __init__(self, dim_mults=(1, 2, 4, 8), out_c=1) -> None:
        super().__init__()

        # useful information
        self.dim_mults = dim_mults
        self.channels = out_c
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)

        # time embedding
        time_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )

        # initial projection
        self.init_conv = nn.Conv2d(out_c, down_channels[0], kernel_size=3, padding=1)        
        # encoder
        self.encoder = Encoder(time_dim, down_channels)
        # decoder
        self.decoder = Decoder(time_dim, up_channels)
        # final layer
        self.out = nn.Conv2d(up_channels[-1], 1, out_c)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)
        x, residuals = self.encoder(x, t)
        x = self.decoder(x, t, residuals)
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
