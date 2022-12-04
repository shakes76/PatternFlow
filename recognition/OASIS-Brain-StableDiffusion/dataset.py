from torch.utils.data import DataLoader
import torchvision



def load_dataset(path, image_size=64, batch_size=64):
    """
    Normalizes and loads images from a specified dataset into a dataloader

    Args:
        path (str): path to the folder containing the dataset
        image_size (int, optional): size, W, of the image (WxW). Defaults to 256.
        batch_size (int, optional): batch size for the dataloader. Defaults to 64.

    Returns:
        DataLoader: pyTorch dataloader of the dataset
    """
    # define the transform used to normalize the input data
    transforms = torchvision.transforms.Compose(
        [
        torchvision.transforms.Resize(image_size+round(0.25*image_size)),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create the pyTorch dataset and dataloader
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transforms)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset_loader
