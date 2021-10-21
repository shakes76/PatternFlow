import matplotlib.pyplot as plt
from data_utils import get_min_imageshape, train_val_test_split
from SegmentationMetrics import dice_coef, dice_loss
from misc_utils import get_close2power
from SegmentaionModel import SegModel


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

    # Get the maximum possible square image shape. 256x256 in this case.
    new_imageshape = get_close2power(min(img_height, img_width))
    print("\nThe maximum possible square image shape is " + str(new_imageshape) + "x" + str(new_imageshape))

    # Load, preprocess and split the data into 60% train, 20% validation and 20% test set
    split_ratio = [0.6, 0.2, 0.2]
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(image_path, mask_path, new_imageshape, new_imageshape, split_ratio, randomstate=42)

    # Construct baseline Unet model
    print("\nConstructing model...")
    model = SegModel((new_imageshape, new_imageshape, 3), random_seed=42, model="Improved_Unet")
    # Test run of the baseline Unet model
    print("Training model...")
    model.train(X_train, X_val, y_train, y_val, optimizer='adam', lr=0.00001, loss=dice_loss, metrics=[dice_coef], batch_size=2, epochs=35)

    # Predict and plot test set
    print("\nPredicting test set...")
    y_pred = model.predict(X_test, batch_size=32)
    print("Plotting First test set and predicted mask...")
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[0])
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(y_test[0, :, :, 0], cmap="gray")
    plt.title("True mask")
    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[0, :, :, 0], cmap="gray")
    plt.title("Predict mask")
    plt.suptitle("First test data")
    plt.tight_layout()
    plt.show()
    print("Plotting Second test set and predicted mask...")
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[1])
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(y_test[1, :, :, 0], cmap="gray")
    plt.title("True mask")
    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[1, :, :, 0], cmap="gray")
    plt.title("Predict mask")
    plt.suptitle("Second test data")
    plt.tight_layout()
    plt.show()

    # Calculate dice coefficient on test set
    print("\nTest set Dice Coefficient:", dice_coef(y_test, y_pred).numpy())


if __name__ == "__main__":
    main()
