__author__ = "Xin Qi"

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

class improved_UNET(object):
    """The improved UNET model for image segmentation"""
    def __init__(self, train_data, validate_data, num_filters=64):
        self._train_data = train_data
        self._vali_data = validate_data
        self._num_filters = num_filters

        self.model = None

    def dice_coef(self, y_true, y_pred, smooth = 1.):
        y_true = tf.cast(y_true, tf.float32)
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1. - self.dice_coef(y_true, y_pred)


    def conv_block(self, input_mat,num_filters,kernel_size,batch_norm):
        X = Conv2D(num_filters,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same')(input_mat)
        if batch_norm:
            X = BatchNormalization()(X)

        X = Activation('relu')(X)

        X = Conv2D(num_filters,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same')(X)
        if batch_norm:
            X = BatchNormalization()(X)

        X = Activation('relu')(X)

        return X

    def Unet(self):
        inputs = Input((256,256,3))
        conv1 = self.conv_block(inputs,self._num_filters,3,batch_norm=True)
        p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
        p1 = Dropout(0.2)(p1)

        conv2 = self.conv_block(p1,self._num_filters*2,3,batch_norm=True)
        p2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
        p2 = Dropout(0.2)(p2)

        conv3 = self.conv_block(p2,self._num_filters*4,3,batch_norm=True)
        p3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
        p3 = Dropout(0.2)(p3)

        conv4 = self.conv_block(p3,self._num_filters*8,3,batch_norm=True)
        p4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)
        p4 = Dropout(0.2)(p4)

        conv5 = self.conv_block(p4,self._num_filters*16,3,batch_norm=True)

        u6 = Conv2DTranspose(self._num_filters*8, (3,3), strides=(2, 2), padding='same')(conv5)
        u6 = concatenate([u6,conv4])
        conv6 = self.conv_block(u6,self._num_filters*8,3,batch_norm=True)
        conv6 = Dropout(0.2)(conv6)

        u7 = Conv2DTranspose(self._num_filters*4,(3,3),strides = (2,2) , padding='same')(conv6)
        u7 = concatenate([u7,conv3])
        conv7 = self.conv_block(u7,self._num_filters*4,3,batch_norm=True)
        conv7 = Dropout(0.2)(conv7)

        u8 = Conv2DTranspose(self._num_filters*2,(3,3),strides = (2,2) , padding='same')(conv7)
        u8 = concatenate([u8,conv2])
        conv8 = self.conv_block(u8,self._num_filters*2,3,batch_norm=True)
        conv8 = Dropout(0.2)(conv8)

        u9 = Conv2DTranspose(self._num_filters,(3,3),strides = (2,2) , padding='same')(conv8)
        u9 = concatenate([u9, conv1])
        conv9 = self.conv_block(u9, self._num_filters, 3, batch_norm=True)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs = inputs, outputs = outputs)

        return model 

    def compile(self, learning_rate = 0.5e-4):
        self.model = self.Unet()
        self.model.compile(optimizer = Adam(lr = learning_rate),
                    loss = self.dice_coef_loss, 
                    metrics = [self.dice_coef])

    def fit(self, batch_size = 8, epoch = 15, checkpoint=True, checkpoint_name = 'ISIC.hdf5'):
        if checkpoint:
            model_checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss',
                                    verbose=1, save_best_only=True)
            self.model.fit(self._train_data.batch(batch_size),
                    validation_data=self._vali_data.batch(batch_size),
                    epochs=epoch, callbacks=[model_checkpoint])

            self.model.load_weights(checkpoint_name)
        else:
            self.model.fit(self._train_data.batch(batch_size),
                    validation_data=self._vali_data.batch(batch_size),
                    epochs=epoch)

        
    def predict(self, raw_images):
        return self.model.predict(raw_images)
    
    def show_result(self, raw_image, ground_truth, save_name = 'result.jpg'):
        predict_result = self.predict(raw_image)

        plt.figure(figsize=(16, 16))
        plt.subplot(1,3,1)
        plt.imshow(tf.reshape(raw_image,(256,256,3)))
        plt.title("raw_image", size=12)
        plt.subplot(1,3,2)
        plt.imshow(tf.reshape(ground_truth,(256,256)), cmap=plt.cm.gray)
        plt.title("ground_truth", size=12)
        plt.subplot(1,3,3)
        plt.imshow(tf.reshape(predict_result,(256,256)), cmap=plt.cm.gray)
        plt.title("predict_image", size=12)

        plt.savefig(save_name)



