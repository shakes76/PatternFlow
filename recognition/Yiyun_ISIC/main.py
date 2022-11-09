"""[Driver script] Main script for running Improved UNet
   Author: Yiyun Zhang (s4513350)
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
IMAGE_HEIGHT = 192  # Height of the input images
IMAGE_WIDTH = 256  # Width of the input images

# Hyperparameters
NUM_CLASSES = 1  # Number of classes for classification
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 10  # Number of epochs for training


def main():
    """Main driver function.
    """
    # load datasets
    print("[1/5] Loading the datasets...")
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
    print("[2/5] Creating Improved UNet model...")
    model = AdvUNet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    print(model.model.summary())

    # compile and train the model
    print("[3/5] Training the model...")
    model.compile()
    history = model.fit(train_dataset, val_dataset,
                        batch_size=BATCH_SIZE, epochs=EPOCHS)

    # evaluate on the test set
    print("[4/5] Evaluating the model...")
    results = model.evaluate(test_dataset, batch_size=BATCH_SIZE)
    print("Test loss:", results[0])
    print("Test accuracy:", results[1])
    print("Test dice coefficient:", results[2])

    # generate plots for predictions and metrics
    print("[5/5] Generating plots...")
    try:
        plot_predictions(model, test_dataset, plot_batch=4, threshold=0.5)
        plot_metrics(history)
        print("Saved plots to ./images/plot.png and ./images/plot_metrics.png")
    except Exception:
        print("There was a problem when saving plots.")


if __name__ == "__main__":
    main()
