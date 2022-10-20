"""
Data loader for loading and preprocessing your data
"""
import os
from torch.utils.data import Dataset, DataLoader
import torchvision


class OASISDataset(Dataset):
    def __init__(self, data_dir):
        self.dir = data_dir
        self.data = os.listdir(data_dir)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = torchvision.io.read_image(f"{self.dir}/{self.data[index]}")
        # image = self.transform(image)
        return image.float()


def get_loaders():
    batch_size = 4
    num_workers = 1

    train_loader = DataLoader(OASISDataset("keras_png_slices_data/keras_png_slices_train"),
                              batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(OASISDataset("keras_png_slices_data/keras_png_slices_test"),
                             batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(OASISDataset("keras_png_slices_data/keras_png_slices_validate"),
                                   batch_size=batch_size, drop_last=True,
                                   num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, validation_loader
