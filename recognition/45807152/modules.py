# Model of Improved UNet
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.layers import Input, Add, concatenate


class ImprovedUNet():
    """
    Improved UNet model for lesion segmentation.

    An improved UNet model based on the "Brain Tumor Segmentation and
    Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge"
    research paper.

    Provides the ability to generate a UNet model for lesion segmentation
    with specified input image dimensions.
    """

    def __init__(self, image_height, image_width, image_channels, filters,
                 kernel_size, alpha=0.01, padding='same', dropout=0.3,
                 upsample=(2, 2)):
        """Initialise new Improved UNet model."""
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.padding = padding
        self.dropout = dropout
        self.upsample = upsample

    def new_conv_layer(self, input, filters, kernel_size, strides):
        """Convolution layer for model."""
        conv_layer = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=self.padding,
                            activation=LeakyReLU(alpha=self.alpha))(input)

        return conv_layer

    def new_context_module(self, input, filters):
        """Context module for model."""
        first_conv_layer = self.new_conv_layer(input, filters,
                                               self.kernel_size, 1)
        second_conv_layer = self.new_conv_layer(first_conv_layer, filters,
                                                self.kernel_size, 1)
        dropout = Dropout(self.dropout)(second_conv_layer)

        return dropout

    def new_upsample_module(self, input, filters):
        """Upsample module for model."""
        upsample_layer = UpSampling2D(self.upsample)(input)
        conv_layer = self.new_conv_layer(upsample_layer, filters,
                                         self.kernel_size, 1)
        return conv_layer

    def new_localisation_module(self, input, filters):
        """Localisation module for model."""
        first_conv_layer = self.new_conv_layer(input, filters,
                                               self.kernel_size, 1)
        second_conv_layer = self.new_conv_layer(first_conv_layer, filters,
                                                self.kernel_size, 1)
        return second_conv_layer

    def new_segmentation_layer(self, input, filters):
        """Segmentation layer for model."""
        conv_layer = self.new_conv_layer(input, filters, (1, 1), 1)
        
        return conv_layer

    def build_model(self, summary=False):
        """
        Build Improved UNet model accoring to U-shaped specifications.

        Model is built from left to right of UNet. Gradient signals from
        the downwards levels (left) are injected into the upwards levels
        (right).
        Print summary if summary is True.
        """
        # Input processing layer
        input_layer = Input(shape=(self.image_height, self.image_width,
                                   self.image_channels))

        # First Level
        # 3x3x3 Convolution (out->16)
        # Context module (out->16)
        first_conv = self.new_conv_layer(input_layer, self.filters,
                                         self.kernel_size, 1)
        first_context = self.new_context_module(first_conv, self.filters)
        first_add = Add()([first_conv, first_context])

        # Second Level
        # 3x3x3 Stride 2 Convolution (out->32)
        # Context module (out->32)
        second_conv = self.new_conv_layer(first_add, self.filters * 2,
                                          self.kernel_size, 2)
        second_context = self.new_context_module(second_conv, self.filters * 2)
        second_add = Add()([second_conv, second_context])

        # Third Level
        # 3x3x3 Stride 2 Convolution (out->64)
        # Context module (out->64)
        third_conv = self.new_conv_layer(second_add, self.filters * 4,
                                         self.kernel_size, 2)
        third_context = self.new_context_module(third_conv, self.filters * 4)
        third_add = Add()([third_conv, third_context])

        # Fourth Level
        # 3x3x3 Stride 2 Convolution (out->128)
        # Context module (out->128)
        fourth_conv = self.new_conv_layer(third_add, self.filters * 8,
                                          self.kernel_size, 2)
        fourth_context = self.new_context_module(fourth_conv, self.filters * 8)
        fourth_add = Add()([fourth_conv, fourth_context])

        # Fifth Level (bottom of UNet)
        # 3x3x3 Stride 2 Convolution (out->256)
        # Context module (out->256)
        # Upsample module (out->128)
        fifth_conv = self.new_conv_layer(fourth_add, self.filters * 16,
                                         self.kernel_size, 2)
        fifth_context = self.new_context_module(fifth_conv, self.filters * 16)
        fifth_add = Add()([fifth_conv, fifth_context])
        fifth_upsample = self.new_upsample_module(fifth_add, self.filters * 8)

        # Now, moving back up the UNet to reconstruct full scale mask
        # Fourth Up Level
        # Inject fourth level gradient signal
        # Localization Module (out->128)
        # Upsample module (out->64)
        fourth_up_concat = concatenate([fifth_upsample, fourth_add])
        fourth_up_local = self.new_localisation_module(fourth_up_concat,
                                                       self.filters * 8)
        fourth_up_upsample = self.new_upsample_module(fourth_up_local,
                                                      self.filters * 4)

        # Third Up Level
        # Inject third level gradient signal
        # Localization Module (out->64)
        # Upsample module (out->32)
        third_up_concat = concatenate([fourth_up_upsample, third_add])
        third_up_local = self.new_localisation_module(third_up_concat,
                                                      self.filters * 4)
        third_up_seg = self.new_segmentation_layer(third_up_local,
                                                   self.filters)
        third_up_seg_up = UpSampling2D()(third_up_seg)
        third_up_upsample = self.new_upsample_module(third_up_local,
                                                     self.filters * 2)

        # Second Up Level
        # Inject second level gradient signal
        # Localization Module (out->32)
        # Upsample module (out->16)
        second_up_concat = concatenate([third_up_upsample, second_add])
        second_up_local = self.new_localisation_module(second_up_concat,
                                                       self.filters * 2)
        second_up_seg = self.new_segmentation_layer(second_up_local,
                                                    self.filters)
        second_up_add = Add()([second_up_seg, third_up_seg_up])
        second_up_seg_up = UpSampling2D()(second_up_add)
        second_up_upsample = self.new_upsample_module(second_up_local,
                                                      self.filters)

        # First Up Level
        # Inject first level gradient signal
        # 3x3x3 Convolution (out->32)
        # Segmentation
        first_up_concat = concatenate([second_up_upsample, first_add])
        first_up_conv = self.new_conv_layer(first_up_concat,
                                            self.filters * 2,
                                            self.kernel_size, 1)
        first_up_seg = self.new_segmentation_layer(first_up_conv,
                                                   self.filters)
        first_up_add = Add()([first_up_seg, second_up_seg_up])

        # Output
        output_layer = Conv2D(2, self.kernel_size, activation="softmax",
                              padding=self.padding)(first_up_add)

        model = Model(inputs=input_layer, outputs=output_layer)
        self.model = model

        if summary:
            print(self.model.summary())
        
        return
