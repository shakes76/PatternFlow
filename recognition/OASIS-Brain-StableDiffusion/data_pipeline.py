from torch.utils.data import DataLoader
import torchvision


def load_dataset(path, image_size=256, batch_size=64):
    """
    Normalizes and loads images from a specified dataset
    into a dataloader

    Args:
        path (str): [description]
        image_size (int, optional): size, W, of the image (WxW). Defaults to 256.
        batch_size (int, optional): batch size for the dataloader. Defaults to 64.

    Returns:
        DataLoader: pyTorch dataloader of the dataset
    """
    transforms = torchvision.transforms.Compose(
        [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x-0.5)*2)
        ]
    )

    dataset = torchvision.datasets.ImageFolder(root=path, transform=transforms)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,)

    return dataset_loader
