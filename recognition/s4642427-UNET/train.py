from dataset import *
from modules import *
import tensorflow.keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

def diceCoefficient(y_true, y_pred):
        """   
            DSC Tensorflow implementation sourced from Medium: 
            [https://medium.com/@karan_jakhar/100-days-of-code-day-7-84e4918cb72c]
            [25/10/2019]
        """
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        
        # Get the pixel intersection of the two images 
        intersection = K.sum(y_true_f * y_pred_f)
        
        # DSC = (2 * intersection) / total_pixels
        diceCoeff = (2. * intersection + 1.) / K.sum(y_true_f) + K.sum(y_pred_f) + 1.)
        
        return diceCoeff
        
        
def diceLoss(y_true, y_pred):
    """
        Defines the dice coefficient loss function, ie 1 - Dice Coefficient.
    """
    return 1 - diceCoefficient(y_true, y_pred)

#TODO need to graph the dice Loss and metric
def train(dset_path, mask_path):
    DL = Dataloader(dset_path, mask_path)
    X_train, Y_train, X_test, Y_test = DL.get_XY_split()

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    model = unet()
    model.compile(optimizer=Adam(), loss=diceLoss, metrics=["accuracy", diceCoefficient])

    # We chose batch_size=16 because google collab screams at my for batch_size = 32
    results = model.fit(X_train, Y_train, batch_size=16, epochs=30,validation_data=(X_test, Y_test), callbacks=callbacks)
    return results, X_test, Y_test