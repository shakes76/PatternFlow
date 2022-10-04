from diffusion_imports import *
from unet_model import *
from diffusion_image_loader import *

BETAS = torch.linspace(0.0001, 0.02, 1000)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, axis=0)


def get_index(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.

    Used in apply noise and de noise
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def apply_noise(image, iteration):
    """
    Adapted from https://huggingface.co/blog/annotated-diffusion
    """
    sqrt_alpha_t = get_index(torch.sqrt(ALPHAS_CUMPROD), iteration, image.shape)
    sqrt_minus_one_alpha = get_index(torch.sqrt(1.0 - ALPHAS_CUMPROD), iteration, image.shape)

    noise = torch.randn_like(image)
    return sqrt_alpha_t.to(0) * image.to(0) + sqrt_minus_one_alpha.to(0) * noise.to(0), noise


def de_noise(img, timestep, model):
    """
    Applies denoising to an image adapted from https://huggingface.co/blog/annotated-diffusion
    """

    alphas_cumprod_prev = F.pad(ALPHAS_CUMPROD[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / ALPHAS)

    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - ALPHAS_CUMPROD)

    sqrt_one_minus_alphas_cumprod_t = get_index(sqrt_one_minus_alphas_cumprod, timestep, img.shape)
    sqrt_recip_alphas_t = get_index(sqrt_recip_alphas, timestep, img.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (img - get_index(BETAS, timestep, img.shape) * model(img, timestep) / sqrt_one_minus_alphas_cumprod_t)

    if timestep == 0:
        return model_mean
    else:
        noise = torch.randn_like(img)
        posterior_variance = BETAS * (1. - alphas_cumprod_prev) / (1. - ALPHAS_CUMPROD)

        return model_mean + torch.sqrt(get_index(posterior_variance, timestep, img.shape)) * noise


def generate_image(model, number = 10):
    model = model.cuda()

    plt.figure(figsize=(15,15))
    rows = number
    column = 10
    counter = 1
    plt.axis("off")
    plt.title("Generated Images Based off Stable Diffusion")
    plt.title("Generated Images Based off Stable Diffusion")

    for row in range(1, rows + 1):
        img = torch.randn((1, 1, 256, 256)).cuda()
        stepsize = int(1000 / column)
        # loop removing noise step by step
        for i in range(999, -1, -1):

            with torch.no_grad():
                img = de_noise(img, torch.tensor([i]).cuda(), model)

            if i % stepsize == 0:
                ax = plt.subplot(rows, column, counter)
                ax.axis('off')
                plt.imshow(img[0].permute(1, 2, 0).detach().cpu())

                counter += 1


    plt.savefig("Generated 10 Images.png")
    plt.show()


def train_model(model):
    batchsize = 4
    data_set = load_data(os.path.join(pathlib.Path(__file__).parent.resolve(), "AKOA_Analysis/"), show=True,
                         batch_size=batchsize)

    model = model.cuda()

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = F.l1_loss

    for epoch in range(10):
        running_loss = 0
        model.train(True)

        for index, data in enumerate(tqdm(data_set)):
            pos = torch.randint(0, 1000, [batchsize]).long()

            data_noisy, noise = apply_noise(data, pos)

            optimizer.zero_grad()

            predicted_noise = model(data_noisy.cuda(), pos.cuda()).cuda()

            loss = criterion(predicted_noise, noise).cuda()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        print(running_loss / len(data_set))

    torch.save(model, "Attempt4")


def main():
    model = torch.load('Attempt3')
    # train_model(model)
    generate_image(model)


if __name__ == "__main__":
    main()
