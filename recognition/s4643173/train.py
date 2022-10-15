import torch
import torch.nn.functional as F

torch.manual_seed(3710)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, model, optimiser):
    for real_images, _ in dataloader:
        real_images = real_images.to(device=DEVICE)

        optimiser.zero_grad()
        vq_loss, recon_image = model(real_images)

        recon_error = F.mse_loss(recon_image, real_images)
        loss = recon_error + vq_loss
        loss.backward()
        optimiser.step()