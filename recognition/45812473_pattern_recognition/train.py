from dataset import process_data, train_val_test_split
from modules import ImprovedUnetModel
from utils import dice_similarity
import tensorflow as tf
import matplotlib.pyplot as plt

def main():

    # Paths to where the data and the masks are located
    data_path = "/content/drive/MyDrive/UQ/all_isic_2016/ISBI2016_ISIC_Part1_Training_Data/ISBI2016_ISIC_Part1_Training_Data/"
    mask_path = "/content/drive/MyDrive/UQ/all_isic_2016/ISBI2016_ISIC_Part1_Training_GroundTruth/ISBI2016_ISIC_Part1_Training_GroundTruth/"

    # Get array for data and corresponding masks
    data, masks = process_data(data_path, mask_path, (128, 128))

    # 80, 10, 10 train val test split
    xtrain, xval, xtest = train_val_test_split(data, 0.8, 0.1)
    ytrain, yval, ytest = train_val_test_split(masks, 0.8, 0.1)

    # Build compile and fit the model
    unet = ImprovedUnetModel()
    unet.build_model()
    unet.model.compile(optimizer = tf.keras.optimizers.Adam(), loss = "binary_crossentropy", metrics = [dice_similarity])
    model = unet.model
    model.summary()
    metrics = model.fit(xtrain, ytrain, epochs = 30, validation_data = (xval, yval), batch_size = 64)

    # Plot the loss and the dice similarity coefficient of the model
    fig, (loss, dsc) = plt.subplots(2, sharex = True)

    loss.set_title("Loss vs Epoch")
    loss.plot(range(len(metrics.history['loss'])), metrics.history['loss'], label = 'Training Loss')
    loss.plot(range(len(metrics.history['val_loss'])), metrics.history['val_loss'], label = 'Validation Loss')
    loss.legend()

    dsc.set_title("Dice Similarity Coefficient vs Epoch")
    dsc.plot(range(len(metrics.history['dice_similarity'])), metrics.history['dice_similarity'], label = "Training Dice Similarity Coefficient")
    dsc.plot(range(len(metrics.history['val_dice_similarity'])), metrics.history['val_dice_similarity'], label = "Validation Dice Similarity Coefficient")
    dsc.legend()

    fig.show()
    fig.savefig("Loss_DSC")




    return


if __name__ == "__main__":
    main()