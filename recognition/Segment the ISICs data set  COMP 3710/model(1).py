import os
from tensorflow import keras
from tensorflow.keras import layers

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