from diffusion_imports import *

class ImageLoader(Dataset):
    def __init__(self, main_dir_slice, main_dir_seg):
        self.main_dir_slice = main_dir_slice
        self.total_imgs_slice = os.listdir(main_dir_slice)

        self.main_dir_seg = main_dir_seg
        self.total_imgs_seg = os.listdir(main_dir_seg)
        self.total_imgs_slice.sort()
        self.total_imgs_seg.sort()

    def __len__(self):
        return len(self.total_imgs_slice)

    def __getitem__(self, idx):
        trans = transforms.ToTensor()

        img_slice = os.path.join(self.main_dir_slice, self.total_imgs_slice[idx])
        slice = trans(Image.open(img_slice).convert("L"))

        img_seg = os.path.join(self.main_dir_seg, self.total_imgs_seg[idx])
        seg = trans(Image.open(img_seg).convert("L"))

        t = transforms.ConvertImageDtype(torch.int8)
        seg = t(seg).div(42, rounding_mode = 'trunc').type(torch.int8)

        return slice, seg