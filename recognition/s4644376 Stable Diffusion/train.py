from diffusion_imports import *
from modules import *
from dataset import *

BETAS = torch.linspace(0.0001, 0.02, 1000)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, axis=0)

PATH_TO_DATASET = 'AKOA_Analysis/'

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


def train_model(model, epochs = 50):
    """
    Main Training Loop
    """
    batchsize = 4
    data_set = load_data(PATH_TO_DATASET, show=True,
                         batch_size=batchsize)

    model = model.cuda()

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = F.l1_loss
    loss_set = []
    validate_loss = []

    for epoch in range(epochs):
        running_loss = 0
        model.train(True)

        # train model on each image with random amounts of noise
        for index, data in enumerate(tqdm(data_set)):
            pos = torch.randint(0, 1000, [batchsize]).long()

            data_noisy, noise = apply_noise(data, pos)

            optimizer.zero_grad()

            predicted_noise = model(data_noisy.cuda(), pos.cuda()).cuda()

            loss = criterion(predicted_noise, noise).cuda()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        loss_set.append([running_loss / len(data_set)])
        print("Loss: ", running_loss / len(data_set))
        validate_model(model, validate_loss)

    torch.save(model, "Rerun2")

    f = open('Loss Data.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(loss_set)

    z = open('Validate Data.csv', 'w')
    writer = csv.writer(z)
    writer.writerows(validate_loss)


def validate_model(model, validate_loss):
    """
    Calculate Validation Loss
    """
    batchsize = 4
    data_set = load_data(PATH_TO_DATASET, show=True,
                         batch_size=batchsize, type="validate")

    model = model.cuda()

    criterion = F.l1_loss

    running_loss = 0

    # loop through and calculate loss for each image with random noise
    for index, data in enumerate(tqdm(data_set)):
        pos = torch.randint(0, 1000, [batchsize]).long()

        data_noisy, noise = apply_noise(data, pos)

        predicted_noise = model(data_noisy.cuda(), pos.cuda()).cuda()

        loss = criterion(predicted_noise, noise).cuda()

        running_loss += loss.item()

    validate_loss.append([running_loss / len(data_set)])
    print("Validation: ", running_loss / len(data_set))


def main():
    if len(sys.argv) > 1:
        train_model(MainNetwork(), int(sys.argv[1]))
    else:
        train_model(MainNetwork())



if __name__ == "__main__":
    main()
