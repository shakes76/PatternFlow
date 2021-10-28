"""Driver script for Improved UNet
"""
from model import AdvUNet
from data import get_filenames, split_data, create_datasets

def main():
    # load datasets
    features, labels = get_filenames(isic_dir="./datasets/ISIC2018")
    train_features, train_labels, val_features, val_labels, test_features, test_labels = split_data(
        features, labels, validation_split=0.2, test_split=0.2)

    # create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_features, train_labels, val_features, val_labels,
        test_features, test_labels, [192, 256], 1)

    for image, label in train_dataset.take(1):
        print("Dataset shapes:", image.shape, label.shape)
        break

    model = AdvUNet()
    print(model.model.summary())
    

if __name__ == "__main__":
    main()
