import matplotlib.pyplot as plt
from data_utils import get_min_imageshape, train_val_test_split
from SegmentationMetrics import dice_coef, dice_loss


def main():
    # Directory path of the images and masks
    image_path = r"C:\Users\masa6\Desktop\UQMasterDS\COMP3710\OpenProject\Project\Data\ISIC2018_Task1-2_Training_Input_x2\*.jpg"
    mask_path = r"C:\Users\masa6\Desktop\UQMasterDS\COMP3710\OpenProject\Project\Data\ISIC2018_Task1_Training_GroundTruth_x2\*.png"

    # Image shapes are not consistent, get the minimum image shape. Shape of [283, 340] in this case.
    print("Getting minimum image shape...")
    min_img_shape = get_min_imageshape(mask_path)
    img_height = min_img_shape[0]
    img_width = min_img_shape[1]
    print("\nMin Image Height:", img_height)
    print("Min Image Width:", img_width)

    # Load, preprocess and split the data into 60% train, 20% validation and 20% test set
    split_ratio = [0.6, 0.2, 0.2]
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(image_path, mask_path, img_height, img_width, split_ratio, randomstate=42)

    # Plot the data
    print("\nPlotting the first preprocessed RGB image and preprocessed mask of the train, validation and test set...")
    # First data in train set
    plt.subplot(3, 2, 1)
    plt.imshow(X_train[0])
    plt.title("First train image")
    plt.subplot(3, 2, 2)
    plt.imshow(y_train[0, :, :, 0], cmap="gray")
    plt.title("First train mask")
    # First data in validation set
    plt.subplot(3, 2, 3)
    plt.imshow(X_val[0])
    plt.title("First validation image")
    plt.subplot(3, 2, 4)
    plt.imshow(y_val[0, :, :, 0], cmap="gray")
    plt.title("First validation mask")
    # First data in test set
    plt.subplot(3, 2, 5)
    plt.imshow(X_test[0])
    plt.title("First test image")
    plt.subplot(3, 2, 6)
    plt.imshow(y_test[0, :, :, 0], cmap="gray")
    plt.title("First test mask")
    plt.tight_layout()
    plt.show()

    # Test the dice coefficient and dice loss function
    # This is expected to be 1
    print("\nDice coefficient between the first train mask:", dice_coef(y_train[0, :, :, 0], y_train[0, :, :, 0]).numpy())
    # This is expected to be 0
    print("Dice loss between the first train mask:", dice_loss(y_train[0, :, :, 0], y_train[0, :, :, 0]).numpy())
    # This is expected to be between 0 and 1, but not 1
    print("Dice coefficient between the first train mask and first test mask:", dice_coef(y_train[0, :, :, 0], y_test[0, :, :, 0]).numpy())
    # This is expected to be between 0 and 1, but not 0
    print("Dice loss between the first train mask and first test mask:", dice_loss(y_train[0, :, :, 0], y_test[0, :, :, 0]).numpy())


if __name__ == "__main__":
    main()
