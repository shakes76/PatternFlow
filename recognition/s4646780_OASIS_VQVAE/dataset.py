import os
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader


class OASISDataset(Dataset):
    """
    Creates the dataset loader for the OASIS dataset.
    Params:
        root_dir: the root directory which contains the dataset.
        transforms: whether the images have any transforms applied to them.
        train: If you want to load the train data.
        validation: If you want to load the validation data.
        test: If you want to load the test data.
    """

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

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image


def create_train_test_loaders():
    """
    Function to create the data loaders for loading and pre-processing the data.
    """
    BATCH_SIZE = 128

    transforms_done = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    train_dataset = OASISDataset(root_dir="./keras_png_slices_data", train=True, transforms=transforms_done)
    test_dataset = OASISDataset(root_dir="./keras_png_slices_data", test=True, transforms=transforms_done)

    # Creates the loaders for train and test data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader
