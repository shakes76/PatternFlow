import torch, torchvision
from dataset import Dataset

if __name__ == '__main__':
    # TODO: optimal batch size?
    batch_size = 6
    dataset_train = Dataset(
        data_path='data/training/data', 
        truth_path='data/training/truth', 
        metadata_path='data/training/data/ISIC-2017_Training_Data_metadata.csv'
        )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)

    print(next(iter(dataloader_train)))

    test = enumerate(dataloader_train)
    batch_no, (data, truth) = next(test)

    print(batch_no)
    print(data.shape)
    print(truth.shape)
