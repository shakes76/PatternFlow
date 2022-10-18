import os

import torch
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


class GANDataset(Dataset):
    """
    Creates the dataset loader for the GAN data.
    Params:
        root_dir: the root directory which contains the dataset.
        transforms: whether the images have any transforms applied to them.
        train: If you want to load the train data.
        validation: If you want to load the validation data.
        test: If you want to load the test data.
    """

    def __init__(self, model, root_dir, transforms=None, train=False, test=False, validation=False):
        self.root_dir = root_dir
        self.vqvae_model = model
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
        image = self.transform(image) if self.transform else image
        image = image.unsqueeze(dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)

        encoded_output = self.vqvae_model.pre_vq_conv(self.vqvae_model.encoder(image))
        _, _, _, encoding = self.vqvae_model.vq_vae(encoded_output)  # gets the encoding from the VQ-VAE to pass to GAN
        encoding = encoding.float().to(device)
        encoding = encoding.view(64, 64)
        encoding = torch.stack((encoding, encoding, encoding), 0)  # GAN uses 3 input channel
        return encoding


def create_train_test_loaders_vqvae():
    """
    Function to create the data loaders for loading and pre-processing the data for VQ-VAE.
    """
    BATCH_SIZE = 128

    transforms_done = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    train_dataset = OASISDataset(root_dir="./keras_png_slices_data", train=True, transforms=transforms_done)
    test_dataset = OASISDataset(root_dir="./keras_png_slices_data", test=True, transforms=transforms_done)

    # Creates the loaders for train and test data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def create_train_test_loaders_dcgan(vqvae_model, GAN_BATCH_SIZE):
    """
    Function to create the data loaders for loading and pre-processing the data for DCGAN.
    """

    transforms_done = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    train_dataset = GANDataset(vqvae_model, root_dir="./keras_png_slices_data", train=True, transforms=transforms_done)
    test_dataset = GANDataset(vqvae_model, root_dir="./keras_png_slices_data", test=True, transforms=transforms_done)

    # Creates the loaders for train and test GAN data
    train_loader_gan = DataLoader(train_dataset, batch_size=GAN_BATCH_SIZE, shuffle=True)
    test_loader_gan = DataLoader(test_dataset, batch_size=GAN_BATCH_SIZE, shuffle=False)
    return train_loader_gan, test_loader_gan
