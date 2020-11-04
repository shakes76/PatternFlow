import os
import cv2
import argparse
import numpy as np
from model import Unet
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, callbacks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def dice_coef(y_true, y_pred, smooth=1):
    '''
    calculate the  Dice similarity coefficient
    :param y_true: the truth value
    :param y_pred: the prediction value
    :return:
    '''
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    '''
    calculate the  Dice similarity coefficient
    :param y_true: the truth value
    :param y_pred: the prediction value
    :return: the Dice similarity coefficient
    '''
    return 1 - dice_coef(y_true, y_pred, smooth=1)

def get_image(data_dir, img_path="Training_Data", mask_path="Training_GroundTruth", size=(256, 256)):
    '''
    obtain the picture
    :param size: the picture size
    :return: the picture
    '''
    X = []
    Y = []
    for i in os.listdir(os.path.join(data_dir, img_path)):
        img = cv2.imread(os.path.join(data_dir, img_path, i)) / 255
        img = cv2.resize(img, size)
        X.append(img)
        mask = cv2.imread(os.path.join(data_dir, mask_path, i[:-4]+"_Segmentation.png"), cv2.IMREAD_GRAYSCALE) / 255
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        mask = cv2.resize(mask, size)
        Y.append(mask)
    return np.array(X), np.array(Y)

def get_db(data_dir):
    '''
    :param data_dir: data
    :return: the split data
    '''
    x, y = get_image(data_dir)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=3)
    Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], 1))
    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "datsets",
                        help="Data set address",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 8,
                        help="number of workers",
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size",
                        )
    parser.add_argument("--epochs",
                        type=int,
                        default = 16,
                        help="number of epochs",
                        )
    parser.add_argument("--lr",
                        type=float,
                        default=0.0001,
                        help="learning rate",
                        )
    parser.add_argument("--logs",
                        type=str,
                        default="./logs",
                        help="Log folder",
                        )
    args = parser.parse_args()

    model = Unet()
    model.build(input_shape=(None, 256, 256, 3))
    model.compile(optimizer=optimizers.Adam(lr=args.lr), 
                loss=dice_coef_loss,
                metrics=['accuracy', dice_coef])
    
    # Loading Dataset
    X_train, Y_train, X_test, Y_test = get_db(args.data_dir)
    # Set tf.keras.callbacks.ModelCheckpoint callback to automatically save the model
    checkpoint_path = "weight/ep{epoch:03d}-val_dice_coef{val_dice_coef:.3f}-val_acc{val_accuracy:.3f}.ckpt"
    modelCheckpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path, # Path to save the model
        verbose=1, # Whether to output information
        save_weights_only=True,
        period=1,# Save the model every few rounds
    )
    earlyStopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Monitored data
    min_delta=0.001, 
    patience=4, # Can accept the number of rounds with a promotion less than min_delta
    )
    # Continuous patience rounds of monotor without improvement will change the learning rate
    reduceLROnPlateau = callbacks.ReduceLROnPlateau(
        factor=0.2, # new_lr = lr * factor
        patience=10, # Can accept rounds without lifting
        min_lr=0.0000000001) # lr lower limit
    tensorboard = callbacks.TensorBoard(
        log_dir=args.logs, 
        write_graph=True, # Visualize images in TensorBoard
        update_freq='epoch'# Write loss and metrics to TensorBoard after each epoch
    )
    model.fit(
        x=X_train,
        y=Y_train,
        epochs=args.epochs,
        validation_data=(X_test, Y_test),
        validation_freq=1, # Test every few rounds
        callbacks=[modelCheckpoint, reduceLROnPlateau, tensorboard, earlyStopping],
        batch_size=args.batch_size,
        workers=args.workers,
    )
