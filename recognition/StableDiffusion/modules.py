from imports import *
from dataset import ImageLoader

"""
Defines a (cosine) beta schedule

Implemented as per the original paper,
also as described in the author's blog post:
https://huggingface.co/blog/annotated-diffusion

The modified cosine scheduler was propsed in the paper:
https://arxiv.org/abs/2102.09672
"""
def beta_schedule(TIMESTEPS, s=0.008):
    steps = TIMESTEPS + 1
    x = torch.linspace(0, TIMESTEPS, steps)
    alphas_cumprod = torch.cos(((x / TIMESTEPS) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    BETAS = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(BETAS, 0.0001, 0.9999)

# Define useful CONSTANTS
TIMESTEPS = 300
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

Mathematically taken from the original paper, also as described in the author's blog post:
https://huggingface.co/blog/annotated-diffusion
"""
def apply_diffusion(x_0, t, device="cuda", noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = extract(SQRT_ALPHAS_CUML_PRODUCT, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(SQRT_ONE_MINUS_ALPHAS_CUML_PRODUCT, t, x_0.shape)

    return sqrt_alphas_cumprod_t * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t * noise.to(device), noise

"""
Return a noisy image, via forward diffusion of a given image
"""
def generate_noisy_image(x_0, t):
    # add noise
    x_noisy, _ = apply_diffusion(x_0, t=t)

    # Reverse of the original applied transform
    reverse_transform = Compose([
                        Lambda(lambda t: (t + 1) / 2),
                        Lambda(lambda t: t.permute(1, 2, 0)),
                        Lambda(lambda t: t * 255.),
                        Lambda(lambda t: t.numpy().astype(np.uint8)),
                        ToPILImage(),
                    ])

    # convert to PIL image
    return reverse_transform(x_noisy.squeeze())

"""
Generate a gif of the forward diffusion process
"""
def generate_sampling_gif(x_0, TIMESTEPS=300):
    import matplotlib.animation as animation
    figure = plt.figure(figsize=(7,7))
    seqn = []
    for t in range(TIMESTEPS):
        image = plt.imshow(generate_noisy_image(x_0, torch.tensor([t])), animated=True)
        seqn.append([image])
    animation = animation.ArtistAnimation(figure, seqn, interval=100, blit=True, repeat_delay=5000)
    animation.save('sampling.gif')

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

Taken from original paper
"""
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        x = torch.log(10000) / (half_dim - 1)
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
The final UNet model
"""
class UNet(nn.Module):
    def __init__(self, dim_mults=(1, 2, 4, 8), channels=1) -> None:
        super().__init__()

        # useful information
        self.dim_mults = dim_mults
        self.channels = channels

        # time embedding
        time_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )

        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)

        # initial projection
        self.init_conv = nn.Conv2d(channels, down_channels[0], kernel_size=3, padding=1)        
        # downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_dim) for i in range(len(down_channels)-1)])
        # upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_dim, up=True) for i in range(len(up_channels)-1)])
        self.out = nn.Conv2d(up_channels[-1], 1, channels)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)
        residuals = []
        for down in self.downs:
            x = down(x,t)
            residuals.append(x)
        for up in self.ups:
            skip = residuals.pop()
            x = torch.cat((x, skip), dim=1)
            x = up(x,t)
        return self.out(x)
