from torchvision import transforms, datasets
import torch 

TRAIN_DATA_PATH = "your/path/to/data"
TEST_DATA_PATH = "your/path/to/data"

def get_data(batch_size_train, batch_size_test):
    """ 
    Loads the train and test data based on the given batch size and normalises 
    the data. 

    Parameters:
        batch_size_train: the batch size for the train set. 
        batch_size_test: the batch size for the test set.

    Returns:
        A tuple containing the train and test data loaders.
    """
    transform = transforms.Compose([
    # Convert the images to tensors
    transforms.ToTensor(),
    # Normalise the images to be between -1 and 1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DATA_PATH, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size_train, shuffle=False,
        num_workers=2, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader