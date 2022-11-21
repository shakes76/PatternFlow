# Run to perform training on Improved UNet model and generate metrics
from dataset import ISIC_Dataset
from modules import ImprovedUNet
from predict import make_prediction
from utility import dice_coefficient, IoU
from config import *

import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


def main():
    """
    Run training on Improved UNet model and generate metrics.
    
    ISIC 2017 lesion data will be used to generate segmentation masks.
    """
    # Image and mask file paths (update relative path where nessesary)
    data_path = "../../../ISIC_Data/ISIC-2017_Training_Data/*.jpg"
    mask_path = "../../../ISIC_Data/ISIC-2017_Training_Part1_GroundTruth/*.png"

    # Load data and split into train, validate and testing subsets
    lesion_data = ISIC_Dataset(data_path, mask_path, IMAGE_HEIGHT, IMAGE_WIDTH)
    train_split = 0.8   # 80% of images will be used for training
    val_split = 0.1     # 10% of images will be used for validation
    test_split = 0.1    # 10% of images will be reseverd for testing
    (train_x, train_y), (val_x, val_y), (test_x, test_y) \
        = lesion_data.get_data_splits(train_split, val_split, test_split)

    # Create UNet model
    model = ImprovedUNet(IMAGE_HEIGHT, IMAGE_WIDTH, image_channels=3,
                         filters=16, kernel_size=(3, 3))
    model.build_model(summary=True)   # Print model summary to terminal

    # Extract UNet model and compile
    model = model.model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="binary_crossentropy",
                  metrics=[dice_coefficient, IoU])

    # Train model and save history
    history = model.fit(x=train_x,
                        y=train_y,
                        epochs=TRAINING_EPOCHS,
                        validation_data=(val_x, val_y),
                        batch_size=64)

    # Save the model locally (tf file)
    model.save("./trainedModel", save_format='tf')

    # Plot training metrics
    # Loss vs Epoch
    plt.figure(0)
    plt.plot(history.history['loss'], label='Loss')
    plt.title("Loss vs Training Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    # Save plot with a tight boarder
    plt.savefig("./figures/lossVsEpoch.png", bbox_inches='tight')

    # Dice Coeficient vs Epoch
    plt.figure(1)
    plt.plot(history.history['dice_coefficient'], label='Dice Coefficient')
    plt.title("Dice Coefficient vs Training Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend(loc='lower right')
    # Save plot with a tight boarder
    plt.savefig("./figures/diceVsEpoch.png", bbox_inches='tight')

    # IoU vs Epoch
    plt.figure(2)
    plt.plot(history.history['IoU'], label='IoU')
    plt.title("IoU vs Training Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Intersection Over Union')
    plt.legend(loc='lower right')
    # Save plot with a tight boarder
    plt.savefig("./figures/iouVsEpoch.png", bbox_inches='tight')

    # Evaluate model performance against test subset
    print(model.evaluate(test_x, test_y, verbose=2, batch_size=64))

    # Utilise test set to make predictions for evaluation
    make_prediction(model, test_x, test_y, slice=(151, 161))

    return


if __name__ == '__main__':
    main()
