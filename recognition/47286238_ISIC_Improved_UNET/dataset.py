import torch, torchvision
from torchvision import transforms as T
import pandas
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, truth_path, metadata_path) -> None:
        self.data_path = data_path
        self.truth_path = truth_path
        self.metadata = pandas.read_csv(metadata_path)

    #     # Load dataset folders
    #     self.folder_train = torchvision.datasets.ImageFolder('data/training', transform=t)
    #     self.folder_test_data = torchvision.datasets.ImageFolder('data/test/data', transform=t)
    #     self.folder_validation_data = torchvision.datasets.ImageFolder('data/validation/data', transform=t)

    #     self.folder_test_truth = torchvision.datasets.ImageFolder('data/test/truth', transform=t)
    #     self.folder_validation_truth = torchvision.datasets.ImageFolder('data/validation/truth', transform=t)

    # def load(self, batch_size) -> None:
    #     self.loader_train = torch.utils.data.DataLoader(self.folder_train, batch_size=batch_size, shuffle=True)
    #     self.loader_test_data = torch.utils.data.DataLoader(self.folder_test_data, batch_size=batch_size)
    #     self.loader_validation_data = torch.utils.data.DataLoader(self.folder_validation_data, batch_size=batch_size)

    #     self.loader_test_truth = torch.utils.data.DataLoader(self.folder_test_truth, batch_size=batch_size)
    #     self.loader_validation_truth = torch.utils.data.DataLoader(self.folder_validation_truth, batch_size=batch_size)

    def __len__(self):
        """
        Reports the size of the dataset
        Required in order to use DataLoader
        """
        len_data = len([file for file in os.listdir(self.data_path) if os.path.isfile(file)])
        return len_data

    def __getitem__(self, key):
        """ 
        Yields a data-groundtruth sample for a given key
        """
        resize = T.Resize((256,256))

        data_id = self.metadata.iloc[key]['image_id']
        data_path = os.path.join(self.data_path, data_id + '.jpg')
        data = torchvision.io.read_image(data_path, mode=torchvision.io.ImageReadMode.RGB)
        # resize and normalize to values between 0 and 1
        data = resize(data)
        data = torch.divide(data, 255)

        truth_path = os.path.join(self.truth_path, data_id + '_segmentation.png')
        truth = torchvision.io.read_image(truth_path, mode=torchvision.io.ImageReadMode.RGB)
        truth = resize(truth)
        truth = torch.divide(truth, 255)

        return data, truth

train = Dataset(data_path='data/training/data', truth_path='data/training/truth', metadata_path='data/training/data/ISIC-2017_Training_Data_metadata.csv')
data, truth = train[0]
print(data.max())
print(data.min())
print(data.shape)
print(truth.max())
print(truth.min())
print(truth.shape)


# testing data dimensions
# FIXME: delete before PR
# ds = Dataset()
# ds.load(12)
# enum = enumerate(ds.loader_train)
# batch_no, (images, labels) = next(enum)
# print(images.min())
# print(images.max())
# print(images.shape)
# print(labels)