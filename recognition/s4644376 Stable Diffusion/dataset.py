from diffusion_imports import *


class ImageLoader(Dataset):
    def __init__(self, path, image_set):
        self.main_dir_slice = path
        self.image_set = image_set

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        trans = transforms.ToTensor()

        img_slice = os.path.join(self.main_dir_slice, self.image_set[idx])

        slice = trans(Image.open(img_slice).convert("L").resize((256, 256)))
        slice = slice.cuda()
        slice = slice.mul(2).sub(1)

        return slice


def load_data(path, type="train", batch_size=32, show=False):
    total_imgs_slice = os.listdir(path)
    total_imgs_slice.sort()

    train = []
    validate = []
    for index, image in enumerate(total_imgs_slice):
        if index % 5 == 0:
            validate.append(image)
        else:
            train.append(image)

    if type == "train":
        loader = ImageLoader(path, train)
    else:
        loader = ImageLoader(path, validate)

    data = DataLoader(loader, batch_size=batch_size, shuffle=True, drop_last=True)

    if show:

        fig = plt.figure(figsize=(10, 7))
        for index, z in enumerate(data):
            x = z

            fig.add_subplot(1, 1, 1)
            plt.imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.title("Reference Image (X)")

            break

    return data
