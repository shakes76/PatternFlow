from diffusion_imports import *
from train import get_index

BETAS = torch.linspace(0.0001, 0.02, 1000)
ALPHAS = 1.0 - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, axis=0)


def de_noise(img, timestep, model):
    """
    Applies denoising to an image adapted from https://huggingface.co/blog/annotated-diffusion
    """

    alphas_cumprod_prev = F.pad(ALPHAS_CUMPROD[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / ALPHAS)

    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - ALPHAS_CUMPROD)

    sqrt_one_minus_alphas_cumprod_t = get_index(sqrt_one_minus_alphas_cumprod, timestep, img.shape)
    sqrt_recip_alphas_t = get_index(sqrt_recip_alphas, timestep, img.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
                img - get_index(BETAS, timestep, img.shape) * model(img, timestep) / sqrt_one_minus_alphas_cumprod_t)

    if timestep == 0:
        return model_mean
    else:
        noise = torch.randn_like(img)
        posterior_variance = BETAS * (1. - alphas_cumprod_prev) / (1. - ALPHAS_CUMPROD)

        return model_mean + torch.sqrt(get_index(posterior_variance, timestep, img.shape)) * noise


def generate_n_image_process(model, number=10):
    """
    Generate n images while showing the diffusion process every 100 noise steps
    """
    model = model.cuda()

    plt.figure(figsize=(15, 15))
    rows = number
    column = 10
    counter = 1
    plt.axis("off")
    plt.title("Generated Images Based off Stable Diffusion")

    # generate images up to specified number
    for row in tqdm(range(1, number + 1)):
        img = torch.randn((1, 1, 256, 256)).cuda()
        stepsize = int(1000 / column)

        # loop removing noise step by step
        for i in tqdm(range(999, -1, -1)):

            with torch.no_grad():
                img = de_noise(img, torch.tensor([i]).cuda(), model)

            if i % stepsize == 0:
                ax = plt.subplot(rows, column, counter)
                ax.axis('off')
                plt.imshow(img[0].permute(1, 2, 0).detach().cpu())

                counter += 1
    plt.savefig("Generated Images Based off Stable Diffusion")
    plt.show()


def generate_single(model):
    """
    Generate a single image with the given model
    """
    model = model.cuda()

    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title("Generated Images Based off Stable Diffusion")

    img = torch.randn((1, 1, 256, 256)).cuda()

    # denoise single image over timestep range of 1000
    for i in tqdm(range(999, -1, -1)):
        with torch.no_grad():
            img = de_noise(img, torch.tensor([i]).cuda(), model)

    plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
    plt.savefig('Single Example')
    plt.show()


def generate_n_images(model, number=10):
    """
    Uses plt to generate n images and lay them in a checkerboard layout
    """
    model = model.cuda()

    plt.figure(figsize=(15, 15))
    rows = number
    column = 10
    counter = 1
    plt.axis("off")
    plt.title("Generated Images Based off Stable Diffusion")

    # generate images up to number specified
    for row in tqdm(range(1, number + 1)):
        img = torch.randn((1, 1, 256, 256)).cuda()

        # loop removing noise step by step
        for i in tqdm(range(999, -1, -1)):
            with torch.no_grad():
                img = de_noise(img, torch.tensor([i]).cuda(), model)

        ax = plt.subplot(rows, column, counter)
        ax.axis('off')
        plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
        counter += 1

    plt.show()


def generate_graphs():
    """
    Generates the graphs used in readme file. Requires csvs formed by train.py
    """
    data_x = []
    data_y = []
    data_y_2 = []

    # Open training loss data and append to list
    with open('Loss Data.csv', newline='') as csvfile:
        csvs = csv.reader(csvfile, delimiter=',', quotechar='|')
        counter = 1
        for row in csvs:
            if row != []:
                data_y.append(float(row[0]))
                data_x.append(counter)
                counter += 1

    # open validate loss data and append to list
    with open('Validate Data.csv', newline='') as a:
        f = csv.reader(a, delimiter=',', quotechar='|')

        for row in f:
            if row != []:
                data_y_2.append(float(row[0]))

    # graph validate and loss data
    plt.plot(data_x, data_y, label='Training Loss')
    plt.plot(data_x, data_y_2, label='Validation Loss')
    plt.title("Validation & Training Loss Across Epochs")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Validation & Training Loss Across Epochs")
    plt.show()


if __name__ == '__main__':
    # try load model based off standard name else ask for path
    try:
        model = torch.load(os.path.join(pathlib.Path(__file__).parent.resolve(), "Stable Diffusion Model OAI OKOA"))
    except Exception as e:
        path = input("Failed to load default model, specify model path: ")
        try:
            model = torch.load(path)
        except Exception:
            print("Failed to load model, verify path and try again")

    # basic error handling but not too great
    if sys.argv[1] == 'single':
        generate_single(model)
    elif sys.argv[1] == 'generate':
        generate_n_images(model, int(sys.argv[2]))
    elif sys.argv[1] == 'illustrate':
        generate_n_image_process(model, int(sys.argv[2]))
    else:
        print("\nInvalid Command! Valid Commands Are:")
        print("\n\tpython predict.py single")
        print("\n\tpython predict.py generate {number_of_images}")
        print("\n\tpython predict.py illustrate {number_of_images}")
