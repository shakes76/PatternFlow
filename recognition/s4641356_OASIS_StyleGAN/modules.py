import tensorflow as tf
import numpy as np
import csv

class adaIN(tf.keras.layers.Layer): #Note to future self, for deterministic layers use keras.layers.Lambda
    """
    Custom Keras Neural Netork layer to conduct adaptive instance normalization
    As specified by StyleGAN.
    """
    def __init__(self, **kwargs) -> None:
        """
        Instantiate new adaIn layer
        """
        super().__init__(**kwargs)

    def build(self, input_shape: np.array) -> None:
        """
        Called automatically on first 'call' passing the shape of input
        To allow weights to be allocated lazily. Validates the applied tensor shape

        Raises:
            ValueError: If the second input tensor does not posess the correct number of scaling values (one per channel)

        Args:
            input (np.array): Shape of input passed to layer
        """
        if not ((input_shape[0][3] == input_shape[1][1]) and (input_shape[0][3] == input_shape[2][1])):
            raise ValueError("Scale tensor must have exacty one value per channel of image input, recieved Image: {}, Scales: {}".format(input_shape[0],input_shape[1]))

        #Each layer posesses a learnt bias
        #self._y_bias = self.add_weight(shape = (1,1,input_shape[0][3]), initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0), name = "channel_adaIN_scales")


    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Performs deterministic adaIN

        Args:
            input (list[tf.Tensor,tf.Tensor, tf.Tensor]): list of tensors, the first is the working image layer, 
                    the second is a tensor containing the coresponding scales for each channel,
                    the third is a tensor containing the corresponding biases for each channel.

        Returns:
            tf.Tensor: image layer scaled and biased (same dimensions as first input tensor)
        """

        x,yscale, ybias = input
        #x will be 4 dimensional, channel last, we conduct fun axis antics to leverage broadcasting
        yscale = yscale[:,tf.newaxis,tf.newaxis,:] 
        ybias = ybias[:,tf.newaxis,tf.newaxis,:]
        mean = tf.math.reduce_mean(tf.math.reduce_mean(x,axis=1),axis=1)[:,tf.newaxis,tf.newaxis,:] 
        std = tf.math.reduce_std(tf.math.reduce_std(x,axis=1),axis=1)[:,tf.newaxis,tf.newaxis,:]

        return (yscale*(x-mean)/std) + ybias

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
        To allow weights to be allocated lazily. Validates the applied tensor shape

        Raises:
            ValueError: If the two input tensors do not have matching dimension

        Args:
            input (np.array): Shape of input passed to layer
        """
        if not (input_shape[0] == input_shape[1]):
            raise ValueError("Inputs must be of same shape, recieved the following: Input 1: {}, Input 2: {}".format(input_shape[0],input_shape[1]))

        #This layer's weights are the scaling of the noise per channel
        self.noise_weight = self.add_weight(shape = (1,1,input_shape[0][3]), initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0), name = "noise_weight") #inherited from Layer

    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Preforms the layer's desired operation using trained weights

        Args:
            input (list[tf.Tensor,tf.Tensor]): List of two Tensors, the first is the current working image layer, and the second is the corresponding matrix of noise to add to it. 
                    The two tensors must have matching dimensions

        Returns:
            tf.Tensor: The image tensor with the noise scaled by learnt weight added to it
        """

        x,noise = input
        return x + (self.noise_weight*noise)

class StyleGAN():
    """
    Class representing a Generational Adversarial Network based off of StyleGAN-1.
    Combines a generator and a discriminator model with custom training paridigm.
    Does not directly subclass keras.Model as this does not allow for unsupervised learning.
    Instead we indirectly call the relevent functionality
    """
    METRICS = ["discrim_loss_real", "discrim_acc_real","discrim_loss_fake", "discrim_acc_fake","gan_loss","gan_acc"] #GAN training metrics 
    GEN_LEARN_RATE = 2.5e-7
    DISCRIM_LEARN_RATE = 3e-8


    def __init__(self, output_res: int = 256, start_res: int = 4, latent_dim: int = 512, existing_model_folder: str = None) -> None:
        """
            Instantiate a new StyleGAN

        Args:
            output_res (int, optional): side length of output images in pixels. Defaults to 256.
            start_res (int, optional): side length of first iamge generation layer. Defaults to 4.
            latent_dim (int, optional): dimension of latent space. Defaults to 512.
            existing_model_folder (str,optional): filepath of existing styleGANModel. If this is not None the other parameters are ignored and existing styleGAN is loaded instead. Defaults to None
        """
        super(StyleGAN, self).__init__()
        #TODO sanitise inputs
        if existing_model_folder is not None:
            self.load_model(existing_model_folder)
        else:
            self._make_model(output_res, start_res, latent_dim, 0, None, None)
            

    def _make_model(self, output_res: int, start_res: int, latent_dim: int, trained_epochs: int, generator: tf.keras.Model, discriminator: tf.keras.Model) -> None: #TODO docstring
        #initialise parameters
        self._output_res = output_res
        self._start_res = start_res
        self._latent_dim = latent_dim

        #initialise generator
        self._generator = generator
        if generator is None:
            self._generator = self.get_generator()
        self._generator_base = tf.ones(shape = (1,self._start_res//2,self._start_res//2,self._latent_dim)) #start with zeros as the background of OASIS images are black, will need repeated for batches

        #initialise discriminator
        self._discriminator = discriminator
        if discriminator is None:
            self._discriminator = self.get_discriminator()
        self._discriminator.trainable = False #enables custom unsupervised training paradigm leveraging keras training efficiency, we just call train on _gan which trains on the output (discrim(gen)) while only actually training the generator
        
        #internal keras model allowing compilation and the assotiated performance benefits during training, takes a latent vector in and returns the discrimination of the generator's output
        self._gan = tf.keras.Model(self._generator.input, self._discriminator(self._generator.output), name = "StyleGAN")
        self._gan.summary()

        #configure components for training
        loss='binary_crossentropy'
        # optimizer = 'adam'
        metrics=['accuracy']
        self._generator.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate=StyleGAN.GEN_LEARN_RATE), metrics = metrics)
        self._gan.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate=StyleGAN.GEN_LEARN_RATE), metrics = metrics)
        self._discriminator.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate=StyleGAN.DISCRIM_LEARN_RATE), metrics = metrics)
       

        self._epochs_trained = trained_epochs


    def get_generator(self) -> tf.keras.Model:
        """
        Creates a generator model inline with the StyleGAN framework        

        Returns:
            tf.keras.Model: uncompiled generator model
        """
        
        generator_latent_input = tf.keras.layers.Input(shape = (self._latent_dim,), name = "Latent_Input")

        #Latent Feature Generation
        z = generator_latent_input
        for i in range(8):
            z = tf.keras.layers.Dense(self._latent_dim, name = "feature_gen_dense_{}".format(i+1))(z)
            z = tf.keras.layers.LeakyReLU(0.2, name = "feature_gen_leakyrelu_{}".format(i+1))(z)

        w = z
        adaIN_scales = tf.keras.layers.Dense(self._latent_dim, name = "adain_scales")(w)
        adaIN_biases = tf.keras.layers.Dense(self._latent_dim, name = "adain_biases")(w)

        #Generation blocks for each feature scale
        curr_res = self._start_res
        
        #using 'tensor =' to give a constant doesn't allow for dynamic batch size (you must specify the length of that dimenstion unless you hack something using a Lambda layer). This will be fixed inside the StyleGan train 
        constant_input = tf.keras.layers.Input(shape = (self._start_res//2,self._start_res//2,self._latent_dim), name = "Constant_Initial_Image") 
       
        x = constant_input
        generator_noise_inputs = [] #keep a hold of input handles for model return
        while curr_res <= self._output_res:
            # filter_num = self._output_res//(curr_res//4)

            #Each resolution needs an appropriately sized noise inputs
            layer_noise_inputs = (tf.keras.layers.Input(shape = (curr_res,curr_res,self._latent_dim), name = "{0}x{0}_Noise_Input_1".format(curr_res)),
                    tf.keras.layers.Input(shape = (curr_res,curr_res,self._latent_dim), name = "{0}x{0}_Noise_Input_2".format(curr_res)))
            generator_noise_inputs += list(layer_noise_inputs)
  
            x = tf.keras.layers.UpSampling2D(size=(2, 2), name = "Upsample_to_{0}x{0}".format(curr_res))(x)
            x = addNoise(name = "{0}x{0}_Noise_1".format(curr_res))([x,layer_noise_inputs[0]])
            x = adaIN(name = "{0}x{0}_adaIN_1".format(curr_res))([x,adaIN_scales,adaIN_biases])
            x = tf.keras.layers.Conv2D(self._latent_dim, kernel_size=3, padding = "same", name = "{0}x{0}_2D_deconvolution".format(curr_res))(x)
            x = addNoise(name = "{0}x{0}_Noise_2".format(curr_res))([x,layer_noise_inputs[1]])
            x = adaIN(name = "{0}x{0}_adaIN_2".format(curr_res))([x,adaIN_scales,adaIN_biases])

            curr_res = curr_res*2
        
        output_image = tf.keras.layers.Conv2D(1, kernel_size=3, padding = "same",name = "Final_Image".format(curr_res))(x)

        return tf.keras.Model(inputs = ([generator_latent_input] + generator_noise_inputs + [constant_input]), outputs = output_image, name = "Generator")
    
    def get_discriminator(self) -> tf.keras.Model:
        """
        Creates a discriminator model inline with the StyleGAN framework        

        Returns:
            tf.keras.Model: uncompiled discriminator model
        """
        discriminator_input = tf.keras.layers.Input(shape = (self._output_res,self._output_res,1), name = "Discriminator_Input") #note we expect greyscale images
        #Feature analysis blocks, perform convolution on decreasing image resolution (and hence filter increasingly macroscopic features)
        current_res = self._output_res
        x = discriminator_input
        while current_res > 4:
            x = tf.keras.layers.Dropout(0.2, name ="{0}x{0}_drop_1".format(current_res))(x)
            x = tf.keras.layers.Conv2D(self._latent_dim*4//current_res,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution_1".format(current_res))(x)
            x = tf.keras.layers.Conv2D(self._latent_dim*4//current_res,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution_2".format(current_res))(x)
            x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_leaky_reLU_1".format(current_res))(x)
            # x = tf.keras.layers.Dropout(0.2, name ="{0}x{0}_drop_2".format(current_res))(x)
            
            # x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_leaky_reLU_2".format(current_res))(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name = "{0}x{0}_image_reduction".format(current_res))(x)
            current_res = current_res//2

        #Flatten and compile features for discrimination
        x = tf.keras.layers.Conv2D(self._latent_dim ,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution_1".format(current_res))(x)
        x = tf.keras.layers.Conv2D(self._latent_dim ,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution_2".format(current_res))(x)
        x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_Leaky_ReLU".format(current_res))(x)
        x = tf.keras.layers.Flatten(name = "Flatten")(x)
        # x = tf.keras.layers.Dense(self._latent_dim, name = "Discriminator_Dense_Classify")(x)
        # x = tf.keras.layers.LeakyReLU(0.2,name = "Flat_Leaky_ReLU")(x)
        
        x = tf.keras.layers.Dense(1, activation = "sigmoid", name = "Discriminate")(x) #Final decision, 1 for real 0 for fake

        return tf.keras.Model(inputs = discriminator_input, outputs = x, name = "Discriminator")
            

    def __call__(self, inputs: list[np.array]) -> np.array:
        """
        Allows the StyleGAN to be used as a functional, takes in a latent vector and an appropriate set of noise matrices, and returns the (normalized) image data of a generated image

        Args:
            inputs (list[np.array]): _description_ TODO

        Returns:
            np.array: _description_ TODO
        """
        return self._generator(inputs + [np.repeat(self._generator_base, inputs[0].shape[0], axis = 0)])

    def save_model(self, folder: str) -> None: #TODO docstring
        #save model parameters
        with open(folder + 'param.csv', mode = 'w', newline='') as f:
            csv.writer(f).writerow([self._output_res,
                    self._start_res,
                    self._latent_dim,
                    self._epochs_trained
                    ]) 

        #save model architecture and assotated weights
        self._discriminator.save(folder + "discriminator") #TODO double check the weights are prorperly being saved in discrim and gen
        self._generator.save(folder + "generator")

    def load_model(self, folder :str) -> None: #TODO docstring
        #Load model Parameters
        params = None
        with open(folder + 'param.csv', mode = 'r') as f:
            params = next(csv.reader(f)) #param csv should be a single row

        #load model architecture and assotiated weights
        discriminator = tf.keras.models.load_model(folder + "discriminator")
        generator = tf.keras.models.load_model(folder + "generator")

        self._make_model(int(params[0]),int(params[1]),int(params[2]),int(params[3]),generator,discriminator)
        print("Successfully found and loaded StyleGAN located in \"{}\"".format(folder))
        
