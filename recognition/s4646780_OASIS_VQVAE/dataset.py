import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2


class OASISDataset(Dataset):
    def __init__(self, root_dir, transforms=None, train=False, test=False, validation=False):
        self.root_dir = root_dir
        if train:
            self.dataset = os.path.join(self.root_dir, "keras_png_slices_train")
        if test:
            self.dataset = os.path.join(self.root_dir, "keras_png_slices_test")
        if validation:
            self.dataset = os.path.join(self.root_dir, "keras_png_slices_validate")
        self.image_paths = [os.path.join(self.dataset, name) for name in os.listdir(self.dataset)]
        self.transform = transforms

    def __len__(self):
        return len([name for name in os.listdir(self.dataset)])

    def __getitem__(self, index):
        return 0

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)

        return image