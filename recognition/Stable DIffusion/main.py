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



if __name__ == "__main__":
    main()
