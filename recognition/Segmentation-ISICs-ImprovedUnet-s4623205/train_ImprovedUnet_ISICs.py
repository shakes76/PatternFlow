"""
Driver script to show train model and prediction example

@author: Jeng-Chung Lien
@student id: 46232050
@email: jengchung.lien@uqconnect.edu.au
"""
import matplotlib.pyplot as plt
from numpy.random import choice
from Modules.data_utils import get_min_imageshape, train_val_test_split
from Modules.SegmentationMetrics import dice_coef, dice_loss
from Modules.misc_utils import get_close2power, image_mask_combiner
from Modules.SegmentaionModel import SegModel


def main():
    # Directory path of the images and masks
    image_path = r"C:\Users\masa6\Desktop\UQMasterDS\COMP3710\OpenProject\Project\Data\ISIC2018_Task1-2_Training_Input_x2\*.jpg"
    mask_path = r"C:\Users\masa6\Desktop\UQMasterDS\COMP3710\OpenProject\Project\Data\ISIC2018_Task1_Training_GroundTruth_x2\*.png"

    # Image shapes are not consistent, get the minimum image shape. Shape of [283, 340] in this case.
    print("Getting minimum image shape...")
    min_img_shape = get_min_imageshape(mask_path)
    img_height = min_img_shape[0]
    img_width = min_img_shape[1]
    print("Min Image Height:", img_height)
    print("Min Image Width:", img_width)

    # Get the maximum possible square image shape. 256x256 in this case.
    new_imageshape = get_close2power(min(img_height, img_width))
    print("\nThe maximum possible square image shape is " + str(new_imageshape) + "x" + str(new_imageshape))

    # Load, preprocess and split the data into 60% train, 20% validation and 20% test set
    split_ratio = [0.6, 0.2, 0.2]
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(image_path, mask_path, new_imageshape, new_imageshape, split_ratio, randomstate=42)

    # Construct Improved Unet model
    print("\nConstructing model...")
    model = SegModel((new_imageshape, new_imageshape, 3), random_seed=42, model="Improved_Unet")
    # Train the Improved Unet model
    print("Training model...")
    model.train(X_train, X_val, y_train, y_val, optimizer='adamW', lr=0.001, loss=dice_loss, metrics=[dice_coef], batch_size=2, epochs=50, lr_decay=True, decay_rate=0.95)

    # Plot the train, validation loss and dice coefficient
    print("\nPlotting train, validation loss and dice coefficient...")
    epoch_range = range(1, len(model.history['loss']) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, model.history['loss'], label='train')
    plt.plot(epoch_range, model.history['val_loss'], label='validation')
    plt.xticks(epoch_range, fontsize=6, rotation=315)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dice Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, model.history['dice_coef'], label='train')
    plt.plot(epoch_range, model.history['val_dice_coef'], label='validation')
    plt.xticks(epoch_range, fontsize=6, rotation=315)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice Coefficient")
    plt.tight_layout()
    plt.show()

    # Predict and plot test set
    print("\nPredicting test set...")
    y_pred = model.predict(X_test, batch_size=32)
    print("Plotting three random test set and predicted mask...")
    indexs = choice(len(X_test), 3, replace=False)
    for i in range(len(indexs)):
        plt.subplot(len(indexs), 4, i * 4 + 1)
        plt.imshow(X_test[indexs[i]])
        plt.title("Image")
        plt.subplot(len(indexs), 4, i * 4 + 2)
        plt.imshow(y_test[indexs[i], :, :, 0], cmap="gray")
        plt.title("True mask")
        plt.subplot(len(indexs), 4, i * 4 + 3)
        plt.imshow(y_pred[indexs[i], :, :, 0], cmap="gray")
        plt.title("Predict mask")
        plt.subplot(len(indexs), 4, i * 4 + 4)
        plt.imshow(image_mask_combiner(X_test[indexs[i]], y_pred[indexs[i], :, :, 0]))
        plt.title("Combined image")
    plt.suptitle("Test set Predictions")
    plt.tight_layout()
    plt.show()

    # Calculate dice coefficient on test set
    print("\nTest set Dice Coefficient:", dice_coef(y_test, y_pred).numpy())


if __name__ == "__main__":
    main()
