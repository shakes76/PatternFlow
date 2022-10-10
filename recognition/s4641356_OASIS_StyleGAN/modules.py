import tensorflow as tf
import numpy as np

class adaIN(tf.keras.layers.Layer):
    """
    Custom Keras Neural Netork layer to conduct adaptive instance normalization
    As specified by StyleGAN. this is deterministic as the learnt scale and bias come from
    an externally trained dense layer.
    """
    def __init__(self, **kwargs) -> None:
        """
        Instantiate new adaIn layer
        """
        super().__init__(**kwargs)

    def build(self, input_shape: np.array) -> None:
        """
        Called automatically on first 'call' passing the shape of input
        To allow weights to be allocated lazily

        Args:
            input (np.array): Shape of input passed to layer
        """
        pass

    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Performs deterministic adaIN

        Args:
            input (list[tf.Tensor,tf.Tensor]): list of tensors, the first is the working image layer, 
                    the second is a (,2) tensor containing the corespondingfeature scale and bias

        Raises:
            ValueError: If the second input tensor is not exactly two values

        Returns:
            tf.Tensor: image layer scaled and biased (same dimensions as first input tensor)
        """

        if not (input[1].shape[1] == 2):
            raise ValueError("Second input must be of shape (,2), recieved {}".format(input[1].shape))

        x,y = input
        yscale,ybias = tf.split(y,2,axis = 1)#axes shifted by 1 to account for batches
        mean = tf.math.reduce_mean(x)
        std = tf.math.reduce_std(x)
        return (yscale[1:0]*(x-mean)/std) + ybias[1:0]

class addNoise(tf.keras.layers.Layer):
    """
    Custom Keras Neural Network layer to add in specified noise scaled by a learnt factor
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Instantiate new addNoise layer
        """
        super().__init__(**kwargs)

    def build(self, input_shape: np.array) -> None:
        """
        Called automatically on first 'call' passing the shape of input
        To allow weights to be allocated lazily

        Args:
            input (np.array): Shape of input passed to layer
        """
        #This layer has a single weight to train, the scaling of the noise
        self.noise_weight = self.add_weight(shape = [1], initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0), name = "noise_weight") #inherited from Layer

    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Preforms the layer's desired operation using trained weights

        Args:
            input (list[tf.Tensor,tf.Tensor]): List of two Tensors, the first is the current working image layer, and the second is the corresponding matrix of noise to add to it. 
                    The two tensors must have matching dimensions

        Raises:
            ValueError: If the two input tensors do not have matching dimension

        Returns:
            tf.Tensor: The image tensor with the noise scaled by learnt weight added to it
        """
        if not (input[0].shape == input[1].shape):
            raise ValueError("Inputs must be of same shape, recieved the following: Input 1: {}, Input 2: {}".format(input[0].shape,input[1].shape))

        x,noise = input
        return x + (self.noise_weight*noise)

class StyleGAN():
    """
    Class representing a Generational Adversarial Network based off of StyleGAN-1.
    Combines a generator and a discriminator model with custom training paridigm.
    Does not directly subclass keras.Model as this does not allow for unsupervised learning.
    Instead we indirectly call the relevent functionality
    """

    def __init__(self, output_res: int = 256, start_res: int = 2, latent_dim: int = 512, filter_count: int = 512) -> None:
        """
            Instantiate a new StyleGAN

        Args:
            output_res (int, optional): side length of output images in pixels. Defaults to 256.
            start_res (int, optional): side length of constant space to begin generation on. Defaults to 2.
            latent_dim (int, optional): dimension of latent space. Defaults to 512.
            filter_count (int, optional): number of filters per convolution layer (will be scaled where appropriate). Defaults to 512
        """
        super(StyleGAN, self).__init__()

        self._output_res = output_res
        self._start_res = start_res
        self._latent_dim = latent_dim
        self._filter_count = filter_count

        self._generator = self.get_generator()
        self._discriminator = self.get_discriminator()

        #internal keras model allowing compilation and the assotiated performance benefits during training, takes a latent vector in and returns the discrimination of the generator's output
        self._gan = tf.keras.Model(self._generator.input, self._discriminator(self._generator.output), name = "StyleGAN")
        self._gan.summary()

    def get_generator(self) -> tf.keras.Model:
        generator_latent_input = tf.keras.layers.Input(shape = (self._latent_dim,), name = "Latent_Input")

        
        #Latent Feature Generation
        z = generator_latent_input
        for i in range(8):
            z = tf.keras.layers.Dense(self._latent_dim, name = "Feature_Gen_Dense_{}".format(i+1))(z)

        w = z
#TODO refactor such that starting resolution is 4 not 2
        #Generation blocks for each feature scale
        curr_res = self._start_res*2
        x = tf.keras.layers.Input(tensor = tf.zeros(shape = (self._start_res,self._start_res)), name = "Constant_Initial_Image")
        generator_noise_inputs = [] #keep a hold of input handles for model return
        while curr_res <= self._output_res:
            #Each resolution needs an appropriately sized noise inputs
            layer_noise_inputs = (tf.keras.layers.Input(shape = (curr_res,curr_res), name = "{0}x{0}_Noise_Input_1".format(curr_res)),
                    tf.keras.layers.Input(shape = (curr_res,curr_res), name = "{0}x{0}_Noise_Input_2".format(curr_res)))
            generator_noise_inputs += list(layer_noise_inputs)

            #Trained feature scaling based off of latent result for adaIN
            adaIn_scales = (tf.keras.layers.Dense(2, name = "{0}x{0}_Feature_Scale_1".format(curr_res))(w),
                    tf.keras.layers.Dense(2, name = "{0}x{0}_Feature_Scale_2".format(curr_res))(w))


            x = tf.keras.layers.UpSampling2D(size=(2, 2), mame = "Upsample_to_{0}x{0}".format(curr_res))
            x = addNoise(name = "{0}x{0}_Noise_1".format(curr_res))(x,generator_noise_inputs[0])
            x = adaIN(name = "{0}x{0}_adaIN_1".format(curr_res))(x,adaIn_scales[0])
            x = tf.keras.layers.Conv2D(self._filter_count, kernel_size=3, padding = "same", name = "{0}x{0}_2D_convolution".format(curr_res))
            x = addNoise(name = "{0}x{0}_Noise_2".format(curr_res))(x,generator_noise_inputs[1])
            x = adaIN(name = "{0}x{0}_adaIN_2".format(curr_res))(x,adaIn_scales[1])
        
        return tf.keras.Model(inputs = ([generator_latent_input] + generator_noise_inputs), outputs = x, name = "Generator")
    
    def get_discriminator(self) -> tf.keras.Model:
        """
            Creates a discriminator model inline with the StyleGAN framework        

        Returns:
            tf.keras.Model: uncompiled discriminator model
        """
        discriminator_input = tf.keras.layers.Input(shape = (self.output_res,self.output_res,1), name = "Discriminator_Input") #note we expect greyscale images
        current_res = self._output_res
        x = discriminator_input
        #Feature analysis blocks, perform convolution on decreasing image resolution (and hence filter increasingly macroscopic features)
        while current_res > 4:
            x = tf.keras.layers.Conv2D(self._filter_count,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution_1".format(current_res))(x)
            x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_Leaky_ReLU_1".format(current_res))(x)
            x = tf.keras.layers.Conv2D(self._filter_count//2,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution_2".format(current_res))(x)
            x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_Leaky_ReLU_2".format(current_res))(x)
            x = tf.keras.layers.AveragePooling2D((2, 2), name = "{0}x{0}_Image_reduction".format(current_res))(x)
            current_res = current_res//2

        #Flatten and compile features for discrimination
        x = tf.keras.layers.Conv2D(self._filter_count,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution".format(current_res))(x)
        x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_Leaky_ReLU".format(current_res))(x)
        x = tf.keras.layers.Flatten(name = "Flatten")(x)
        x = tf.keras.layers.Dense(self._filter_count, name = "Discriminator_Dense_Classify")(x)
        x = tf.keras.layers.LeakyReLU(0.2,name = "Flat_Leaky_ReLU")(x)
        
        x = tf.keras.layers.Dense(1, activation = "sigmoid", name = "Discriminate")(x) #Final decision, 1 for real 0 for fake

        return tf.keras.Model(inputs = discriminator_input, outputs = x, name = "Discriminator")

    # TODO def __call__():