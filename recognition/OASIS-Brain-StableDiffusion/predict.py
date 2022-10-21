import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import PIL
import numpy as np
from modules import *
from utils import *


def show_single_image(model):
    """
    Create and show a single image from the model

    Args:
        model (Module): PyTorch model to use
    """
    model = model.to("cuda")

    # Set image to random noise
    image = torch.randn((1, 3, 64, 64)).to("cuda")

    # Iteratively remove the noise
    for i in tqdm(range(999, -1, -1)):
        with torch.no_grad():
            image = remove_noise(image, torch.tensor([i]).to("cuda"), model)

    # convert image back into range [0, 255]
    image = image[0].permute(1, 2, 0).detach().to("cpu")
    image = image*255
    image = np.array(image, dtype=np.uint8)
    
    # Get unique time stamp
    dt = datetime.now()
    ts = round(datetime.timestamp(dt))

    # Display using pyplot
    plt.figure(figsize=(4,4))
    plt.axis("off")
    plt.title("Image Created from Stable Diffusion Model")
    plt.imshow(image)
    plt.savefig('Image{}.png'.format(ts))
    plt.show()


def load_model(model_path):
    """
    Load a PyTorch model from a saved model file

    Args:
        model_path (string): file path for the model

    Returns:
        Module: model PyTorch Module
    """
    model = UNet()
    # model.load_state_dict(torch.load(model_path))
    torch.load(model_path)
    model.to("cuda")
    model.eval()
    return model

def main():
    model_path = r"D:\COMP3710\rangpur\train2\DDPM_Uncondtional\ckpt.pt"
    model = load_model(model_path)
    show_single_image(model)


if __name__ == "__main__":
    main()