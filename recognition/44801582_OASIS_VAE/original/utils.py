import torch


def gen_samples(images, model, device):
    with torch.no_grad():
        images = images.to(device)
        reconstruction, _, _ = model(images)

    return reconstruction
