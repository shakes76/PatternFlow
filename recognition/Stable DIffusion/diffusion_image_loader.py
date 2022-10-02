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
        slice = trans(Image.open(img_slice).convert("L").resize((250, 250)))
        slice = slice.mul(2).sub(1)

        return slice