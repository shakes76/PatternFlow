import torch, torchvision
from torchvision import transforms

class Dataset:
    def __init__(self) -> None:
        # Resize images into a uniform shape
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

        # Load dataset folders
        self.folder_train_data = torchvision.datasets.ImageFolder('data/training/data', transform=t)
        self.folder_test_data = torchvision.datasets.ImageFolder('data/test/data', transform=t)
        self.folder_validation_data = torchvision.datasets.ImageFolder('data/validation/data', transform=t)

        self.folder_train_truth = torchvision.datasets.ImageFolder('data/training/truth', transform=t)
        self.folder_test_truth = torchvision.datasets.ImageFolder('data/test/truth', transform=t)
        self.folder_validation_truth = torchvision.datasets.ImageFolder('data/validation/truth', transform=t)

    def load(self, batch_size) -> None:
        self.loader_train_data = torch.utils.data.DataLoader(self.folder_train_data, batch_size=batch_size)

# testing data dimensions
# FIXME: delete before PR
ds = Dataset()
ds.load(24)
enum = enumerate(ds.loader_train_data)
batch_no, (images, labels) = next(enum)
print(images.min())
print(images.max())
print(images.shape)