"""Driver script for Improved UNet
"""
from model import AdvUNet
from data import get_filenames, split_data, create_datasets

# The root path of the datasets.
# Please put the following two folders:
# -  ISIC2018_Task1_Training_GroundTruth and
# -  ISIC2018_Task1-2_Training_Input
# under this root path:
ISIC_DIR = "./datasets/ISIC2018"

# height, width
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256
NUM_CLASSES = 1


def main():
    # load datasets
    features, labels = get_filenames(isic_dir=ISIC_DIR)
    train_features, train_labels, val_features, val_labels, test_features, test_labels = split_data(
        features, labels, validation_split=0.2, test_split=0.2)

    # create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_features, train_labels, val_features, val_labels,
        test_features, test_labels, [IMAGE_HEIGHT, IMAGE_WIDTH], NUM_CLASSES)

    for image, label in train_dataset.take(1):
        print("Dataset shapes:", image.shape, label.shape)
        break

    # model = AdvUNet()


if __name__ == "__main__":
    main()
