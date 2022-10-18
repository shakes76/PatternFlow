import tensorflow as tf
import tensorflow_addons as tfa


# from tensorflow import keras
# from tensorflow.keras import layers


class ImprovedUNET(tf.keras.Model):

    def __init__(self):
        super(ImprovedUNET, self).__init__()
        self.padding = "same"
        self.initial_output = 16
        self.contextDropoutRate = 0.3
        self.leakyAlpha = 0.01

    def context_module(self, input, output_filters):
        convolution1 = tfa.layers.InstanceNormalization()(input)
        convolution1 = tf.keras.layers.Conv2D(output_filters, kernel_size=(3, 3), padding=self.padding,
                                              activation='relu')(convolution1)
        dropout = tf.keras.layers.Dropout(self.contextDropoutRate)(convolution1)
        convolution2 = tfa.layers.InstanceNormalization()(dropout)
        convolution2 = tf.keras.layers.Conv2D(output_filters, kernel_size=(3, 3), padding=self.padding,
                                              activation='relu')(convolution2)
        return convolution2

    def perform_upsampling(self, input, output_filters):
        upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(input)
        upsample = tf.keras.layers.Conv2D(output_filters, kernel_size=(3, 3), padding=self.padding,
                                          activation='relu')(upsample)
        upsample = tfa.layers.InstanceNormalization()(upsample)
        return upsample

    def localization_module(self, input, output_filters):
        convolution1 = tfa.layers.InstanceNormalization()(input)
        convolution1 = tf.keras.layers.Conv2D(output_filters, kernel_size=(3, 3), padding=self.padding,
                                              activation='relu')(convolution1)
        convolution2 = tfa.layers.InstanceNormalization()(convolution1)
        convolution2 = tf.keras.layers.Conv2D(output_filters, kernel_size=(3, 3), padding=self.padding,
                                              activation='relu')(convolution2)
        return convolution2

    def data_pipe_line(self):
        input = tf.keras.layers.Input(shape=(256, 256, 3))

        # Encoder

        convolution1 = tf.keras.layers.Conv2D(self.initial_output, kernel_size=(3, 3), padding=self.padding,
                                              activation='relu')(input)
        convolution_module1 = self.context_module(convolution1, self.initial_output)
        sum1 = tf.keras.layers.Add()([convolution1, convolution_module1])
        first_skip = sum1

        convolution2 = tf.keras.layers.Conv2D(self.initial_output * 2, kernel_size=(3, 3), strides=(2, 2),
                                              padding=self.padding, activation='relu')(sum1)
        convolution_module2 = self.context_module(convolution2, self.initial_output * 2)
        sum2 = tf.keras.layers.Add()([convolution2, convolution_module2])
        second_skip = sum2

        convolution3 = tf.keras.layers.Conv2D(self.initial_output * 4, kernel_size=(3, 3), strides=(2, 2),
                                              padding=self.padding, activation='relu')(sum2)
        convolution_module3 = self.context_module(convolution3, self.initial_output * 4)
        sum3 = tf.keras.layers.Add()([convolution3, convolution_module3])
        third_skip = sum3

        convolution4 = tf.keras.layers.Conv2D(self.initial_output * 8, kernel_size=(3, 3), strides=(2, 2),
                                              padding=self.padding, activation='relu')(sum3)
        convolution_module4 = self.context_module(convolution4, self.initial_output * 8)
        sum4 = tf.keras.layers.Add()([convolution4, convolution_module4])
        fourth_skip = sum4

        convolution5 = tf.keras.layers.Conv2D(self.initial_output * 16, kernel_size=(3, 3), strides=(2, 2),
                                              padding=self.padding, activation='relu')(sum4)
        convolution_module5 = self.context_module(convolution5, self.initial_output * 16)
        sum5 = tf.keras.layers.Add()([convolution5, convolution_module5])

        # Decoder
        upsample_module1 = self.perform_upsampling(sum5, self.initial_output * 8)
        concatenation_module1 = tf.keras.layers.concatenate([upsample_module1, fourth_skip])
        localisation_output1 = self.localization_module(concatenation_module1, self.initial_output * 8)

        upsample_module2 = self.perform_upsampling(localisation_output1, self.initial_output * 4)
        concatenation_module2 = tf.keras.layers.concatenate([upsample_module2, third_skip])
        localisation_output2 = self.localization_module(concatenation_module2, self.initial_output * 4)

        lower_segmented = tf.keras.layers.Conv2D(1, (1, 1), padding=self.padding)(localisation_output2)
        upscaled_lower_segment = tf.keras.layers.UpSampling2D(size=(2, 2))(lower_segmented)

        upsample_module3 = self.perform_upsampling(localisation_output2, self.initial_output * 2)
        concatenation_module3 = tf.keras.layers.concatenate([upsample_module3, second_skip])
        localisation_output3 = self.localization_module(concatenation_module3, self.initial_output * 2)

        middle_segmented = tf.keras.layers.Conv2D(1, (1, 1), padding=self.padding)(localisation_output3)
        first_skip_sum = tf.keras.layers.Add()([upscaled_lower_segment, middle_segmented])
        upscaled_middle_segment = tf.keras.layers.UpSampling2D(size=(2, 2))(first_skip_sum)

        upsample_module4 = self.perform_upsampling(localisation_output3, self.initial_output * 1)
        concatenation_module4 = tf.keras.layers.concatenate([upsample_module4, first_skip])

        convolution_module6 = tf.keras.layers.Conv2D(self.initial_output * 2, kernel_size=(3, 3), padding=self.padding)(
            concatenation_module4)
        upper_segmented = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding=self.padding)(convolution_module6)
        final_node = tf.keras.layers.Add()([upscaled_middle_segment, upper_segmented])

        activation = tf.keras.layers.Activation("sigmoid")(final_node)

        model = tf.keras.Model(inputs=input, outputs=activation)

        return model