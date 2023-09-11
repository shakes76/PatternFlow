import torch
DEVICE = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")


def generate_samples(model, img_dim):
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().to(DEVICE)

    x_tilde = model.generate(label, shape=(img_dim, img_dim), batch_size=100)

    print(x_tilde[0])
