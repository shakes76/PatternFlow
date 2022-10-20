"""
Source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training.
"""
from dataset import *
from modules import VQ_VAE
from utils import gen_samples

import torch
from torch.nn.functional import mse_loss
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
from datetime import datetime


def train(model, data, optimizer, writer, iter):
    for images in data:
        images = images.to(device)

        optimizer.zero_grad()
        reconstruct, encoding, x_quantized = model(images)

        loss_recons = mse_loss(reconstruct, images)
        loss_vq = mse_loss(x_quantized, encoding.detach())
        loss_commit = mse_loss(encoding, x_quantized.detach())

        loss = loss_recons + loss_vq + beta * loss_commit
        loss.backward()

        writer.add_scalar('train_reconstruction', loss_recons.item(), iter)
        writer.add_scalar('train_quantization', loss_vq.item(), iter)

        optimizer.step()


def test(model, data, writer, iter):
    with torch.no_grad():
        reconstruction_loss, vector_quant_loss = 0., 0.
        for images in data:
            images = images.to(device)
            reconstruct, encoding, x_quantized = model(images)
            reconstruction_loss += mse_loss(reconstruct, images)
            vector_quant_loss += mse_loss(x_quantized, encoding)

        reconstruction_loss = (reconstruction_loss/len(data))
        vector_quant_loss = (vector_quant_loss/len(data))

    writer.add_scalar('test_reconstruction', reconstruction_loss, iter)
    writer.add_scalar('test_quantization', vector_quant_loss, iter)


def run_epoch(model, train_data, test_data, optimizer, writer, iter):
    print(f"training epoch {iter}")
    train(model, train_data, optimizer, writer, iter)
    print(f"testing epoch {iter}")
    test(model, test_data, writer, iter)


def main():
    writer = SummaryWriter(f"./logs/{current_time}")

    train_loader, test_loader, validation_loader = get_loaders()

    model = VQ_VAE(256, 512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fixed_images = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    for epoch in range(3):
        run_epoch(model, train_loader, test_loader, optimizer, writer, epoch)

        with open(f"{save_filename}/model_{epoch + 1}.pt", 'wb') as f:
            torch.save(model.state_dict(), f)

        reconstruction = gen_samples(fixed_images, model, device)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('reconstruction', grid, epoch + 1)


if __name__ == "__main__":
    lr = 2e-4
    beta = 1

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else 'cpu')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    save_filename = f"./models/{current_time}"
    os.makedirs(save_filename)

    main()
