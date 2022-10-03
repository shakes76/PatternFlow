from diffusion_imports import *

class ImageLoader(Dataset):
    def __init__(self, path):
        self.main_dir_slice = path
        self.total_imgs_slice = os.listdir(path)
        self.total_imgs_slice.sort()

    def __len__(self):
        return len(self.total_imgs_slice)

    def __getitem__(self, idx):
        trans = transforms.ToTensor()

        img_slice = os.path.join(self.main_dir_slice, self.total_imgs_slice[idx])
        slice = trans(Image.open(img_slice).convert("L").resize((256, 256)))
        slice = slice.cuda()
        slice = slice.mul(2).sub(1)

        return slice


def load_data(path, batch_size=32, workers=None, show=False):
    loader = ImageLoader(path)
    data = DataLoader(loader, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)

    if show:
        fig = plt.figure(figsize=(10, 7))
        for index, z in enumerate(data):
            x = z

            fig.add_subplot(1, 1, 1)
            plt.imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.title("Reference Image (X)")

            break

    return data
