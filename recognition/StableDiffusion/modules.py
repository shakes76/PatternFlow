from imports import *
from dataset import ImageLoader

"""
Defines a (linear) beta schedule

The specific beta values are defined as per the original paper,
also as described in the author's blog post:
https://huggingface.co/blog/annotated-diffusion
"""
def beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end   = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# definitions
device = "cuda" if torch.cuda.is_available() else "cpu"
timesteps = 300
betas = beta_schedule(timesteps)
alphas = 1. - betas
alphas_cuml_product = torch.cumprod(alphas, axis=0)
alphas_cuml_product_prev = F.pad(alphas_cuml_product[:-1], (1, 0), value=1.)
sqrt_recip_alphas = torch.sqrt(1. / alphas)
sqrt_alphas_cuml_product = torch.sqrt(alphas_cuml_product)
sqrt_one_minus_alphas_cuml_product = torch.sqrt(1. - alphas_cuml_product)
posterior_variance = betas * (1. - alphas_cuml_product_prev) / (1. - alphas_cuml_product)

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

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cuml_product, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cuml_product, t, x_0.shape)

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
Show a given image
"""
def generate_image(x_0, t):
    # Reverse of the original applied transform
    reverse_transform = Compose([
                            Lambda(lambda t: (t + 1) / 2),
                            Lambda(lambda t: t * 255.),
                            Lambda(lambda t: t.numpy().astype(np.uint8)),
                            ToPILImage(),
                    ])
    plt.imshow(reverse_transform(x_0))

"""
Generate a gif of the forward diffusion process
"""
def generate_sampling_gif(x_0, timesteps=300):
    import matplotlib.animation as animation
    figure = plt.figure(figsize=(7,7))
    seqn = []
    for t in range(timesteps):
        image = plt.imshow(generate_noisy_image(x_0, torch.tensor([t])), animated=True)
        seqn.append([image])
    animation = animation.ArtistAnimation(figure, seqn, interval=100, blit=True, repeat_delay=5000)
    animation.save('sampling.gif')

"""
Load in image(s) and transform data via custom ImageLoader, and create our DataLoader
"""
def load_dataset(batch_size=8, image_resize=256):
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
    train_ad_imgs = ImageLoader("ADNI_DATA/AD_NC/train/AD",
                                transform=transform)
    train_nc_imgs = ImageLoader("ADNI_DATA/AD_NC/train/NC",
                                transform=transform)
    # Don't need an explicit test set with our model, so may as well
    # combine our test and train datasets to obtain more samples to train with
    total_imgs = torch.utils.data.ConcatDataset([train_ad_imgs, train_nc_imgs])
    return DataLoader(total_imgs, batch_size=batch_size, shuffle=False, num_workers=1)
