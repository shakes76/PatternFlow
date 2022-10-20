import torch

EPOCHS = 2
DEVICE = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")


def train_vqvae(dl, model, optim):
    losses = []
    for batch, (x, _) in enumerate(dl):
        x = x.to(DEVICE)

        optim.zero_grad()
        vq_loss, data_recon = model(x)
        recon_error = torch.nn.functional.mse_loss(data_recon, x) / 0.0338
        loss = recon_error + vq_loss
        loss.backward()
        optim.step()
        losses.append(recon_error.item())
        if batch % 25 == 0:
            print(f"batch {batch:4}/{len(dl)} \t |current loss: {recon_error.item():.6f}")

    losses = sum(losses) / len(losses)
    return losses
