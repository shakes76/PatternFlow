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
        add1 = layers.Add()([convolution1, convolution_module1])
        first_skip = add1

        convolution2 = layers.Conv2D(self.initial_output * 2, (3, 3), padding=self.padding)(input_layer)
        convolution_module2 = self.context_module(convolution1, self.initial_output * 2)
        add2 = layers.Add()([convolution2, convolution_module2])
        second_skip = add2

        convolution3 = layers.Conv2D(self.initial_output * 4, (3, 3), padding=self.padding)(input_layer)
        convolution_module3 = self.context_module(convolution1, self.initial_output * 4)
        add3 = layers.Add()([convolution3, convolution_module3])
        third_skip = add3

        convolution4 = layers.Conv2D(self.initial_output * 8, (3, 3), padding=self.padding)(input_layer)
        convolution_module4 = self.context_module(convolution1, self.initial_output * 8)
        add4 = layers.Add()([convolution4, convolution_module4])
        fourth_skip = add4

        convolution5 = layers.Conv2D(self.initial_output * 16, (3, 3), padding=self.padding)(input_layer)
        convolution_module5 = self.context_module(convolution1, self.initial_output * 16)
        add5 = layers.Add()([convolution5, convolution_module5])
        fifth_skip = add5





