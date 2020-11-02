import os
import cv2
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, callbacks, layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Unet(keras.Model):
    def __init__(self):
        '''
        Build the main UNET model
        There are 9 main layers in total. Different main layers contain several basic layers.
        '''
        super(Unet, self).__init__()
        # 1
        # This main layers contain 2 Convolutional layers and 1 Pooling layer.
        self.conv1_1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv1_1")
        self.conv1_2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv1_2")
        self.down_pool1 = layers.MaxPool2D(pool_size=(2, 2), name="down_pool1")

        # 2
        # This main layers contain 2 Convolutional layers and 1 Pooling layer.
        self.conv2_1 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2_1")
        self.conv2_2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2_2")
        self.down_pool2 = layers.MaxPool2D(pool_size=(2, 2), name="down_pool2")

        # 3
        # This main layers contain 2 Convolutional layers and 1 Pooling layer.
        self.conv3_1 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv3_1")
        self.conv3_2 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv3_2")
        self.down_pool3 = layers.MaxPool2D(pool_size=(2, 2), name="down_pool3")

        # 4
        # This main layers contain 2 Convolutional layers and 1 Pooling layer.
        self.conv4_1 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv4_1")
        self.conv4_2 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv4_2")
        self.down_pool4 = layers.MaxPool2D(pool_size=(2, 2), name="down_pool4")

        # 5
        # This main layers contain 2 Convolutional layers.
        self.conv5_1 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv5_1")
        self.conv5_2 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv5_2")

        # 6
        # up-conv
        # This main layers contains 5 basic layers. 1 upsampling layer, 4 Convolutional layer.
        self.upsampling6 = layers.UpSampling2D(size=(2, 2), name="upsampling6")
        self.up_conv6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="up_conv6")
        self.merge6 = layers.Concatenate(axis=3, name="merge6")
        self.improve_conv6_1 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="improve_conv6_1")

        self.conv6_1 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv6_1")
        self.conv6_2 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv6_2")

        # 7
        # This main layers contains 5 basic layers. 1 upsampling layer, 4 Convolutional layer.
        self.upsampling7 = layers.UpSampling2D(size=(2, 2), name="upsampling7")
        self.up_conv7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="up_conv7")
        self.merge7 = layers.Concatenate(axis=3, name="merge7")
        self.improve_conv7_1 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="improve_conv7_1")

        self.conv7_1 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv7_1")
        self.conv7_2 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv7_2")

        # 8
        # This main layers contains 5 basic layers. 1 upsampling layer, 4 Convolutional layer.
        self.upsampling8 = layers.UpSampling2D(size=(2, 2), name="upsampling8")
        self.up_conv8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="up_conv8")
        self.merge8 = layers.Concatenate(axis=3, name="merge8")
        self.improve_conv8_1 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="improve_conv8_1")

        self.conv8_1 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv8_1")
        self.conv8_2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv8_2")

        # 9
        # This main layers contains 7 basic layers. 1 upsampling layer, 6 Convolutional layer.
        self.upsampling9 = layers.UpSampling2D(size=(2, 2), name="upsampling9")
        self.up_conv9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="up_conv9")
        self.merge9 = layers.Concatenate(axis=3, name="merge9")
        self.improve_conv9_1 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="improve_conv9_1")

        self.conv9_1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv9_1")
        self.conv9_2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv9_2")
        # conv 1x1
        self.conv9_3 = layers.Conv2D(2, 1, activation='relu', padding='same', kernel_initializer='he_normal', name="conv9_3")
        # out
        self.out = layers.Conv2D(1, 1, activation='sigmoid', name="out")

    def call(self, input, training=False):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        down_pool1 = self.down_pool1(conv1_2)

        conv2_1 = self.conv2_1(down_pool1)
        conv2_2 = self.conv2_2(conv2_1)
        down_pool2 = self.down_pool2(conv2_2)

        conv3_1 = self.conv3_1(down_pool2)
        conv3_2 = self.conv3_2(conv3_1)
        down_pool3 = self.down_pool3(conv3_2)

        conv4_1 = self.conv4_1(down_pool3)
        conv4_2 = self.conv4_2(conv4_1)
        # drop4 = self.drop4(conv4_2)
        down_pool4 = self.down_pool4(conv4_2)

        conv5_1 = self.conv5_1(down_pool4)
        conv5_2 = self.conv5_2(conv5_1)
        # drop5 = self.drop5(conv5_2)

        upsampling6 = self.upsampling6(conv5_2)
        up_conv6 = self.up_conv6(upsampling6)
        improve_conv6_1 = self.improve_conv6_1(conv4_2)
        merge6 = self.merge6([improve_conv6_1, up_conv6])
        conv6_1 = self.conv6_1(merge6)
        conv6_2 = self.conv6_2(conv6_1)

        upsampling7 = self.upsampling7(conv6_2)
        up_conv7 = self.up_conv7(upsampling7)
        improve_conv7_1 = self.improve_conv7_1(conv3_2)
        merge7 = self.merge7([improve_conv7_1, up_conv7])
        conv7_1 = self.conv7_1(merge7)
        conv7_2 = self.conv7_2(conv7_1)

        upsampling8 = self.upsampling8(conv7_2)
        up_conv8 = self.up_conv8(upsampling8)
        improve_conv8_1 = self.improve_conv8_1(conv2_2)
        merge8 = self.merge8([improve_conv8_1, up_conv8])
        conv8_1 = self.conv8_1(merge8)
        conv8_2 = self.conv8_2(conv8_1)

        upsampling9 = self.upsampling9(conv8_2)
        up_conv9 = self.up_conv9(upsampling9)
        improve_conv9_1 = self.improve_conv9_1(conv1_2)
        merge9 = self.merge9([improve_conv9_1, up_conv9])
        conv9_1 = self.conv9_1(merge9)
        conv9_2 = self.conv9_2(conv9_1)
        conv9_3 = self.conv9_3(conv9_2)
        out = self.out(conv9_3)
        return out

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
