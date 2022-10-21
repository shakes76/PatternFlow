import torch
import numpy as np

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


def train_pixel_cnn(model, dl, criterion, n_embeddings, optimiser):
    train_loss = []
    for batch_idx, (x, label) in enumerate(dl):

        x = (x[:, 0]).to(DEVICE)

        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, n_embeddings),
            x.view(-1)
        )

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss.append(loss.item())

        if batch_idx % 25 == 0:
            print(f"Batch {batch_idx * len(x)}/{len(dl.dataset)} \tLoss: {loss.item()}")


def test_pixel_cnn(model, dl, criterion, n_embeddings):
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(dl):
            x = (x[:, 0]).to(DEVICE)

            logits = model(x, label)

            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, n_embeddings),
                x.view(-1)
            )

            val_loss.append(loss.item())

    print(f"Validation Completed!\tLoss: {np.asarray(val_loss).mean(0)}")
    return np.asarray(val_loss).mean(0)
