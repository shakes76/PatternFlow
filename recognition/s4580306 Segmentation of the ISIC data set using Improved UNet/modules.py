import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


class ImprovedUNET(tf.keras.Model):

    def __init__(self):
        super(ImprovedUNET, self).__init__()
        self.padding = "same"
        self.initial_output = 16
        self.contextDropoutRate = 0.3
        self.leakyAlpha = 0.01

    def context_module(self, input, output_filters):
        convolution1 = tfa.layers.InstanceNormalization()(input)
        convolution1 = layers.Conv2D(output_filters, (3, 3), padding=self.padding,
                                     activation=layers.leakyReLU(alpha=self.leakyAlpha))(convolution1)
        dropout = layers.Dropout(self.contextDropoutRate)(convolution1)
        convolution2 = tfa.layers.InstanceNormalization()(dropout)
        convolution2 = layers.Conv2D(output_filters, (3, 3), padding=self.padding,
                                     activation=layers.leakyReLU(alpha=self.leakyAlpha))(convolution2)
        return convolution2

    def perform_upsampling(self, input, output_filters):
        upsample = layers.UpSampling2D((2, 2))(input)
        upsample = layers.Conv2D(output_filters, (3, 3), padding=self.padding,
                                 activation=layers.LeakyReLU(alpha=self.leakyAlpha))(upsample)
        upsample = tfa.layers.InstanceNormalization()(upsample)
        return upsample

    def localization_module(self, input, output_filters):
        convolution1 = tfa.layers.InstanceNormalization()(input)
        convolution1 = layers.Conv2D(output_filters, (3, 3), padding=self.padding,
                                     activation=layers.LeakyReLU(alpha=self.leakyAlpha))(convolution1)
        convolution2 = tfa.layers.InstanceNormalization()(convolution1)
        convolution2 = layers.Conv2D(output_filters, (3, 3), padding=self.padding,
                                     activation=layers.LeakyReLU(alpha=self.leakyAlpha))(convolution2)
        return convolution2

    def data_pipe_line(self):

        input_layer = layers.Input(shape=(256, 256, 3))

        # Encoder

        convolution1 = layers.Conv2D(self.initial_output, (3, 3), padding=self.padding)(input_layer)
        convolution_module1 = self.context_module(convolution1, self.initial_output)
        sum1 = layers.Add()([convolution1, convolution_module1])
        first_skip = sum1

        convolution2 = layers.Conv2D(self.initial_output * 2, (3, 3), padding=self.padding)(input_layer)
        convolution_module2 = self.context_module(convolution1, self.initial_output * 2)
        sum2 = layers.Add()([convolution2, convolution_module2])
        second_skip = sum2

        convolution3 = layers.Conv2D(self.initial_output * 4, (3, 3), padding=self.padding)(input_layer)
        convolution_module3 = self.context_module(convolution1, self.initial_output * 4)
        sum3 = layers.Add()([convolution3, convolution_module3])
        third_skip = sum3

        convolution4 = layers.Conv2D(self.initial_output * 8, (3, 3), padding=self.padding)(input_layer)
        convolution_module4 = self.context_module(convolution1, self.initial_output * 8)
        sum4 = layers.Add()([convolution4, convolution_module4])
        fourth_skip = sum4

        convolution5 = layers.Conv2D(self.initial_output * 16, (3, 3), padding=self.padding)(input_layer)
        convolution_module5 = self.context_module(convolution1, self.initial_output * 16)
        sum5 = layers.Add()([convolution5, convolution_module5])

        # Decoder
        upsample_module1 = self.perform_upsampling(sum5, self.initial_output * 8)
        concatenation_module1 = layers.concatenate([upsample_module1, fourth_skip])
        localisation_output1 = self.localization_module(concatenation_module1, self.initial_output * 8)

        upsample_module2 = self.perform_upsampling(localisation_output1, self.initial_output * 4)
        concatenation_module2 = layers.concatenate([upsample_module2, third_skip])
        localisation_output2 = self.localization_module(concatenation_module2, self.initial_output * 4)

        segmentation_one = layers.Conv2D(1, (1, 1), padding=self.padding)(localisation_output2)
        segmentation_one = layers.UpSampling2D(size=(2, 2))(segmentation_one)

        upsample_module3 = self.perform_upsampling(localisation_output2, self.initial_output * 2)
        concatenation_module3 = layers.concatenate([upsample_module3, fourth_skip])
        localisation_output3 = self.localization_module(concatenation_module3, self.initial_output * 2)

        segmentation_two = layers.Conv2D(1, (1, 1), padding=self.padding)(localisation_output3)
        sum6 = layers.Add()([segmentation_one, segmentation_two])
        sum6 = layers.UpSampling2D(size=(2, 2))(sum6)

        upsample_module4 = self.perform_upsampling(localisation_output3, self.initial_output * 1)
        concatenation_module4 = layers.concatenate([upsample_module4, fourth_skip])

        convolution_module6 = layers.Conv2D(self.initial_output * 2, (3, 3), padding=self.padding)(concatenation_module4)
        segmentation_three = layers.Conv2D(1, (1, 1), padding=self.padding)(convolution_module6)
        sum7 = layers.Add()([sum, segmentation_three])

        outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(sum7)
        model = tf.keras.Model(inputs=input, outputs=outputs)

        return model






