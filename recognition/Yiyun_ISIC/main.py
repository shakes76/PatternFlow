"""Driver script for Improved UNet
"""
from model import AdvUNet
from data import get_filenames, split_data, create_datasets
from plots import plot_metrics, plot_predictions

# The root path of the datasets.
# Please put the following two folders:
# -  ISIC2018_Task1_Training_GroundTruth and
# -  ISIC2018_Task1-2_Training_Input
# under this root path:
ISIC_DIR = "./datasets"
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

# hyperparameters
NUM_CLASSES = 1
BATCH_SIZE = 32
EPOCHS = 20


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

    # create model
    model = AdvUNet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    print(model.model.summary())

    # compile and train the model
    model.compile()
    history = model.fit(train_dataset, val_dataset,
                        batch_size=BATCH_SIZE, epochs=EPOCHS)

    # evaluate on the test set
    results = model.evaluate(test_dataset, batch_size=BATCH_SIZE)
    print("Test loss:", results[0])
    print("Test accuracy:", results[1])
    print("Test dice coefficient:", results[2])

    # generate plots for predictions and metrics
    plot_predictions(model, test_dataset, plot_batch=4, threshold=0.5)
    plot_metrics(history)


if __name__ == "__main__":
    main()
