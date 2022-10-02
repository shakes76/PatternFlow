import torch

from diffusion_imports import *
from unet_model import *
from diffusion_image_loader import *


def load_data(path, batch_size=4, show=False):
    loader = ImageLoader(path)
    data = DataLoader(loader, batch_size=batch_size, shuffle=True)

    if show:
        fig = plt.figure(figsize=(10, 7))
        for index, data in enumerate(data):
            x = data

            fig.add_subplot(1, 1, 1)
            plt.imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.title("Reference Image (X)")

            break

    return data

def apply_noise(image, iteration):

    alpha = torch.cumprod(1.0 - torch.linspace(0.0001, 0.02, 1000), axis=0)
    sqrt_alpha_t = torch.sqrt(alpha)[iteration]
    sqrt_minus_one_alpha = torch.sqrt(1. - alpha)[iteration]

    noise = torch.randn_like(image)

    return sqrt_alpha_t.to(0) * image.to(0) + sqrt_minus_one_alpha.to(0) * noise.to(0)




def main():
    torch.set_printoptions(profile="full")
    data = load_data(os.path.join(pathlib.Path(__file__).parent.resolve(), "AKOA_Analysis/"), show=True)

    num_images = 10
    stepsize = int(300 / num_images)


    for image in data:
        break

    fig = plt.figure(figsize=(10, 7))
    print(image.shape)
    for idx in range(0, 300, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int((idx/stepsize) + 1))
        image = apply_noise(image, t)
        print(image.shape)
        plt.imshow(image[3].cpu())
    plt.show()





if __name__ == "__main__":
    main()
