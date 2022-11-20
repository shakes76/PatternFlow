from dataset import *
from modules import *
import tensorflow.keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

def diceCoefficient(y_true, y_pred):

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    # Get the pixel intersection of the two images 
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    
    # DSC = (2 * intersection) / total_pixels
    diceCoeff = (2. * intersection + 1.) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1.)
    
    return diceCoeff
    
    
def diceLoss(y_true, y_pred):
  
    return 1 - diceCoefficient(y_true, y_pred)


def train(dset_path, mask_path):
    DL = Dataloader(dset_path, mask_path)
    X_train, Y_train, X_test, Y_test, X_val, Y_val = DL.get_XY_split()

    # We want our trainer to lower the learning rate when it starts slowing down, and to save the best weights
    callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

    model = Unet()
    model.compile(optimizer=Adam(), loss=diceLoss, metrics=["accuracy", diceCoefficient])

    results = model.fit(X_train, Y_train, batch_size=16, epochs=25,validation_data=(X_val, Y_val), callbacks=callbacks)
    return model, results, X_val, Y_val