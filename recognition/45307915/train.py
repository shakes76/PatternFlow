import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import modules
import dataset

TRAIN_IMAGES_PATH = "./ISIC-2017_Training_Data/*.jpg"
TRAIN_MASKS_PATH = "./ISIC-2017_Training_Part1_GroundTruth/*.png"

TEST_IMAGES_PATH = "./ISIC-2017_Test_v2_Data/*.jpg"
TEST_MASKS_PATH = "./ISIC-2017_Test_v2_Part1_GroundTruth/*.png"

VALIDATE_IMAGES_PATH = "./ISIC-2017_Validation_Data/*.jpg"
VALIDATE_MASKS_PATH = "./ISIC-2017_Validation_Part1_GroundTruth/*.png"

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

BATCH_SIZE = 32

INIT_LEARNING_RATE = 5e-4
EPOCHS = 20

def batchData(train, test, validate):
    """
    Batches the training, testing and validation datasets

    Parameters:
        train (tf.Dataset): A (img_height, img_width, 1) tensor containing the training data
        test (tf.Dataset): A (img_height, img_width, 1) tensor containing the testing data
        validate (tf.Dataset): A (img_height, img_width, 1) tensor containing the validation data

    Return:
        (tf.Dataset, tf.Dataset, tf.Dataset): 3 (img_height, img_width, 1) tensors containing ????

    """
    train_batch = train.batch(BATCH_SIZE)
    test_batch = test.batch(BATCH_SIZE)
    validate_batch = validate.batch(BATCH_SIZE)
    return train_batch, test_batch, validate_batch

def diceCoefficient(y_true, y_pred):
    """
    Dice Coefficient

    Parameters:
        y_true (tf.Tensor): true output
        y_true (tf.Tensor): output predicted by model

    Return:
        tf.Tensor: Dice coefficient based on true output and prediction

    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_true_f))
    return dice

def diceLoss(y_true, y_pred):
    """
    Dice loss function

    Parameters:
        y_true (tf.Tensor): true output
        y_true (tf.Tensor): output predicted by model

    Return:
        tf.Tensor: Dice coefficient to be used as a loss function 

    """
    return 1 - diceCoefficient(y_true, y_pred)

def main():

    # Data loading and preprocessing
    dataLoader = DataLoader()
    train_dataset, test_dataset, validate_dataset = dataLoader.loadData()
    
    train_batched, test_batched, validate_batched = batchData(train_dataset, test_dataset, validate_dataset)
    
    # Generate the model
    improvedUNETModel = ImprovedUNETModel()
    model = improvedUNETModel.modelArchitecture()
    
    adamOptimizer = Adam(learning_rate=INIT_LEARNING_RATE)
    model.compile(optimizer=adamOptimizer, loss=diceLoss, metrics=[diceCoefficient])
    
    results = model.fit(train_batched, epochs=EPOCHS, validation_data=validate_batched)


if __name__ == "__main__":
    main()