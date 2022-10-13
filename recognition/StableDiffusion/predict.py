from imports import *
from dataset import ImageLoader
from modules import *

plt.style.use('ggplot')

"""
Return a slightly less denoised copy of a given image via our UNet network
"""
@torch.no_grad()
def reverse_diffusion_step(model, x, t, t_index):
    betas_t = extract(BETAS, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(SQRT_ONE_MINUS_ALPHAS_CUML_PRODUCT, t, x.shape)
    sqrt_recip_alphas_t = extract(SQRT_RECIP_ALPHAS, t, x.shape)
    
    # use model to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_var_t = extract(POSTERIOR_VAR, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

"""
Reverse the diffusion process
"""
@torch.no_grad()
def reverse_diffusion(model, shape=(1, 1, 256, 256), plot="None"):
    device = next(model.parameters()).device

    if plot != "None":
        fig = plt.figure(figsize=(15,15))
        plt.axis("off")

    reverse_transform = Compose([
                    Lambda(lambda t: (t + 1) / 2),
                    Lambda(lambda t: t * 255.),
                ])

    # sample noise from a Gaussian distribution
    img = torch.randn(shape, device=device)
    imgs = []

    if plot == "plot_diffusion_process":
        rows = 5
        num_images = 5
        stepsize = int(TIMESTEPS/num_images)
        counter = 1
    elif plot == "image_grid":
        rows = 2
        cols = 2
        stepsize = int(TIMESTEPS/1)
        counter = 1
    elif plot == "image_gif":
        rows = 1
        cols = 1
        stepsize = int(TIMESTEPS/100)
        # need to keep track of all images when making gif
        ims = []

    # use our network to gradually denoise the noisy image
    # and save a copy of this progress every "stepsize" steps
    for i in range(0, TIMESTEPS)[::-1]:
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = reverse_diffusion_step(model, img, t, i)
        if i % stepsize == 0:
            if plot == "plot_diffusion_process":
                ax = plt.subplot(rows,num_images,counter)
                ax.axis("off")
                plt.imshow(reverse_transform(img[0]).permute(1,2,0).detach().cpu(), cmap="gray")
                counter+=1
            elif plot == "image_grid":
                ax = plt.subplot(rows,cols,counter)
                ax.axis("off")
                plt.imshow(reverse_transform(img[0].detach().cpu()), cmap="gray")
                counter+=1
            elif plot == "image_gif":
                im = plt.imshow(reverse_transform(img[0]).permute(1,2,0).detach().cpu(), cmap="gray", animated=True)
                ims.append([im])
            else:
                imgs.append(reverse_transform(img).permute(1,2,0).detach().cpu(), cmap="gray")
    return imgs

if __name__ == "__main__":
    model = torch.load("DiffusionModel")
    """
    Predict options include:
    -> 'plot_diffusion_process'
    -> 'image_grid'
    -> 'image_gif'
    """
    reverse_diffusion(model, "image_grid")