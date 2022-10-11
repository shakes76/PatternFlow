from multiprocessing.sharedctypes import Value
import tensorflow as tf
import numpy as np
import GANutils

class adaIN(tf.keras.layers.Layer): #Note to future self, for deterministic layers use keras.layers.Lambda
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
        To allow weights to be allocated lazily. Validates the applied tensor shape

        Raises:
            ValueError: If the second input tensor is not exactly two values, or if the two inputs have differing channel depths

        Args:
            input (np.array): Shape of input passed to layer
        """
        if not (input_shape[1][1] == 2):
            raise ValueError("Second input must be of shape (,2,), recieved {}".format(input_shape[1]))
        if not (input_shape[0][3] == input_shape[1][2]):
            raise ValueError("Inputs must have same number of channels (trailing dimension), recieved Input1: {}, Input 2: {}".format(input_shape[0],input_shape[1]))

    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Performs deterministic adaIN

        Args:
            input (list[tf.Tensor,tf.Tensor]): list of tensors, the first is the working image layer, 
                    the second is a (,2) tensor containing the corespondingfeature scale and bias

        Returns:
            tf.Tensor: image layer scaled and biased (same dimensions as first input tensor)
        """

        x,y = input
        yscale,ybias = tf.split(y,2,axis = 1)#axes shifted by 1 to account for batches
        yscale,ybias = yscale[:,:,tf.newaxis,:],ybias[:,:,tf.newaxis,:]#x will be 4 dimensional, channel last, we conduct fun axis antics to leverage broadcasting
        mean = tf.math.reduce_mean(tf.math.reduce_mean(x,axis=1),axis=1)[:,tf.newaxis,tf.newaxis,:] 
        std = tf.math.reduce_std(tf.math.reduce_std(x,axis=1),axis=1)[:,tf.newaxis,tf.newaxis,:]

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
        To allow weights to be allocated lazily. Validates the applied tensor shape

        Raises:
            ValueError: If the two input tensors do not have matching dimension

        Args:
            input (np.array): Shape of input passed to layer
        """
        if not (input_shape[0] == input_shape[1]):
            raise ValueError("Inputs must be of same shape, recieved the following: Input 1: {}, Input 2: {}".format(input_shape[0],input_shape[1]))

        #This layer has a single weight to train, the scaling of the noise
        self.noise_weight = self.add_weight(shape = [1], initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0), name = "noise_weight") #inherited from Layer

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

    def __init__(self, output_res: int = 256, start_res: int = 4, latent_dim: int = 512, existing_model_filepath: str = None) -> None:
        """
            Instantiate a new StyleGAN

        Args:
            output_res (int, optional): side length of output images in pixels. Defaults to 256.
            start_res (int, optional): side length of constant space to begin generation on. Defaults to 2.
            latent_dim (int, optional): dimension of latent space. Defaults to 512.
            existing_model_filepath (str,optional): filepath of existing styleGANModel. If this is not None the other parameters are ignored and existing styleGAN is loaded instead. Defaults to None
        """
        super(StyleGAN, self).__init__()
        #TODO sanitise inputs
        self._output_res = output_res
        self._start_res = start_res
        self._latent_dim = latent_dim

        self._generator = self.get_generator()
        self._discriminator = self.get_discriminator()

        self._discriminator.trainable = False #enables custom unsupervised training paradigm leveraging keras training efficiency, we just call train on _gan which trains on the output (discrim(gen)) while only actually training the generator

        #internal keras model allowing compilation and the assotiated performance benefits during training, takes a latent vector in and returns the discrimination of the generator's output
        self._gan = tf.keras.Model(self._generator.input, self._discriminator(self._generator.output), name = "StyleGAN")
        self._gan.summary()

        self._gan_metrics = None

        self._epochs_trained = 0

    def get_generator(self) -> tf.keras.Model:

        

        generator_latent_input = tf.keras.layers.Input(shape = (self._latent_dim,), name = "Latent_Input")

        
        #Latent Feature Generation
        z = generator_latent_input
        for i in range(8):
            z = tf.keras.layers.Dense(self._latent_dim, name = "Feature_Gen_Dense_{}".format(i+1))(z)

        w = z

#TODO refactor such that starting resolution is 4 not 2
        #Generation blocks for each feature scale
        curr_res = self._start_res
        #using tensor = to give a constant doesn't allow for dynamic batch size (you must specify the length of that dimenstion without using a Lambda layer). This will be fixed inside the StyleGan train 
        constant_input = tf.keras.layers.Input(shape = (self._start_res//2,self._start_res//2,self._latent_dim), name = "Constant_Initial_Image") 
        x = constant_input
        generator_noise_inputs = [] #keep a hold of input handles for model return
        while curr_res <= self._output_res:
            #Each resolution needs an appropriately sized noise inputs
            layer_noise_inputs = (tf.keras.layers.Input(shape = (curr_res,curr_res,self._latent_dim), name = "{0}x{0}_Noise_Input_1".format(curr_res)),
                    tf.keras.layers.Input(shape = (curr_res,curr_res,self._latent_dim), name = "{0}x{0}_Noise_Input_2".format(curr_res)))
            generator_noise_inputs += list(layer_noise_inputs)

            #Trained feature scaling based off of latent result for adaIN TODO perhaps do simple split
            adaIn_scales = []
            for i in range(self._latent_dim):
                adaIn_scales.append((tf.keras.layers.Dense(2, name = "{0}x{0}_Channel_{1}_Feature_Scale_1".format(curr_res,i+1))(w),
                    tf.keras.layers.Dense(2, name = "{0}x{0}_Channel_{1}_Feature_Scale_2".format(curr_res,i+1))(w)))
            concat_adaIn_scales = (tf.keras.layers.Concatenate(axis = 2, name = "{0}x{0}_Feature_Scales_1".format(curr_res,i+1))([channel[0][:,:,tf.newaxis] for channel in adaIn_scales]),
                    tf.keras.layers.Concatenate(axis = 2, name = "{0}x{0}_Feature_Scales_2".format(curr_res,i+1))([channel[1][:,:,tf.newaxis] for channel in adaIn_scales]))

            x = tf.keras.layers.UpSampling2D(size=(2, 2), name = "Upsample_to_{0}x{0}".format(curr_res))(x)
            x = addNoise(name = "{0}x{0}_Noise_1".format(curr_res))([x,layer_noise_inputs[0]])
            x = adaIN(name = "{0}x{0}_adaIN_1".format(curr_res))([x,concat_adaIn_scales[0]])
            x = tf.keras.layers.Conv2D(self._latent_dim, kernel_size=3, padding = "same", name = "{0}x{0}_2D_convolution".format(curr_res))(x)
            x = addNoise(name = "{0}x{0}_Noise_2".format(curr_res))([x,layer_noise_inputs[1]])
            x = adaIN(name = "{0}x{0}_adaIN_2".format(curr_res))([x,concat_adaIn_scales[1]])

            curr_res = curr_res*2
        
        output_image = tf.keras.layers.Conv2D(1, kernel_size=3, padding = "same", name = "Final_Image".format(curr_res))(x)

        return tf.keras.Model(inputs = ([generator_latent_input] + generator_noise_inputs + [constant_input]), outputs = output_image, name = "Generator")
    
    def get_discriminator(self) -> tf.keras.Model:
        """
            Creates a discriminator model inline with the StyleGAN framework        

        Returns:
            tf.keras.Model: uncompiled discriminator model
        """
        discriminator_input = tf.keras.layers.Input(shape = (self._output_res,self._output_res,1), name = "Discriminator_Input") #note we expect greyscale images
        current_res = self._output_res
        x = discriminator_input
        #Feature analysis blocks, perform convolution on decreasing image resolution (and hence filter increasingly macroscopic features)
        while current_res > 4:
            x = tf.keras.layers.Conv2D(self._latent_dim,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution_1".format(current_res))(x)
            x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_Leaky_ReLU_1".format(current_res))(x)
            x = tf.keras.layers.Conv2D(self._latent_dim,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution_2".format(current_res))(x)
            x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_Leaky_ReLU_2".format(current_res))(x)
            x = tf.keras.layers.AveragePooling2D((2, 2), name = "{0}x{0}_Image_reduction".format(current_res))(x)
            current_res = current_res//2

        #Flatten and compile features for discrimination
        x = tf.keras.layers.Conv2D(self._latent_dim,kernel_size=3,padding = "same", name = "{0}x{0}_2D_convolution".format(current_res))(x)
        x = tf.keras.layers.LeakyReLU(0.2,name = "{0}x{0}_Leaky_ReLU".format(current_res))(x)
        x = tf.keras.layers.Flatten(name = "Flatten")(x)
        x = tf.keras.layers.Dense(self._latent_dim, name = "Discriminator_Dense_Classify")(x)
        x = tf.keras.layers.LeakyReLU(0.2,name = "Flat_Leaky_ReLU")(x)
        
        x = tf.keras.layers.Dense(1, activation = "sigmoid", name = "Discriminate")(x) #Final decision, 1 for real 0 for fake

        return tf.keras.Model(inputs = discriminator_input, outputs = x, name = "Discriminator")

    def compile(self, loss: str, optimizer: str, metrics: list[str], **kwargs) -> None:
        """
        Compiles GAN with specified training metrics, enables training

        Args:
            loss (str): keras string name of loss function to use during training (can also directly pass in a function handle)
            optimizer (str): keras string name of optimizer to use during training
            metrics (list[str]): list of keras string names for metrics that you wish to track during training
        """ #TODO hard code some of these
        self._gan_metrics = metrics
        self._epochs_trained = 0 #recompiling signifies we want to start a new training paradigm
        self._generator.compile(loss, optimizer, metrics, **kwargs)
        self._discriminator.compile(loss, optimizer, metrics, **kwargs)
        self._gan.compile(loss, optimizer, metrics, **kwargs)

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """
        Get model's current training metrics

        Returns:
            list[tf.keras.metrics.Metric]: list of current training metrics
        """
        return self._gan_metrics

    def fit(self, images: np.array, batch_size: int, epochs: int, training_history_location: str, image_sample_count:int = 5, image_sample_output:str = "output/", model_output:str = "model") -> None:
        """
            Fits the model to a given set of training images. The custom GAN training paradigm is as follows: TODO
        Args:
            images (np.array): vector of normalized image data to fit GAN to
            batch_size (int): number of images to pass through model per batch
            epochs (int): number of epochs (repeat runs of full training set) to train model on
            training_history_location (str): filepath of csv (no file extension) to save training history
            image_sample_count (int, optional): Number of image samples to take per epoch ignored if image_sample_output is set to None. Defaults to 5
            image_sample_output (str, optional): Directory to save sample images after each epoch, or set to None to disable saving. Defaults to "output/". TODO Go back to all folder args and rework to not require trailing /
            model_output (str, optional): directory save model information after each epoch (NO TRAILING SLASH), or set to None to disable model saving. Defaults to "model". 
        """
        #TODO sanitse inputs
        num_batches = images.shape[0]//batch_size #required batches per epoch
        batches = np.split(images[:num_batches*batch_size,:,:,:],num_batches,axis = 0) + [images[num_batches*batch_size:,:,:,:]] #split requires an exact even split, so append remainder manually
        
        #Manual Labels to assign to batches when conducting supervised training on discriminator
        real_labels = tf.ones(shape = (batch_size,))
        fake_labels = tf.zeros(shape = (batch_size,))

        #start with zeros as the background of OASIS images are black
        constant_generator_base = tf.zeros(shape = (batch_size,self._start_res//2,self._start_res//2,self._latent_dim))
        
        #Conduct training
        for e in range(epochs):
            print(">> epoch {}/{}".format(e+1,epochs))
            epoch_metrics = {metric: [] for metric in StyleGAN.METRICS}
            for b in range(num_batches):
                print(">>>> batch {}/{}".format(b+1,num_batches))

                #train discriminator on real images
                dfl, dfa = self._discriminator.train_on_batch(batches[b], real_labels)

                #train discriminator on a batch of fake images from current iteration of the generator
                fake_images = self._generator(GANutils.random_generator_inputs(batch_size,self._latent_dim,self._start_res,self._output_res) + [constant_generator_base])
                drl, dra = self._discriminator.train_on_batch(fake_images, fake_labels)

                #Train generator. Indirect training of discriminator weights are disabled, so we train the generator weights to make something that outputs 'real' (1) from discriminator
                gl, ga = self._gan.train_on_batch(GANutils.random_generator_inputs(batch_size,self._latent_dim,self._start_res,self._output_res) + [constant_generator_base], real_labels)

                metrics_to_store = [dfl,dfa,drl,dra,gl,ga]
                for m in range(len(StyleGAN.METRICS)):
                    epoch_metrics[StyleGAN.METRICS[m]].append(metrics_to_store[m])

            self._epochs_trained += 1
            GANutils.save_training_history(epoch_metrics,training_history_location)
            
            if image_sample_output:
                samples = self(GANutils.random_generator_inputs(image_sample_count,self._latent_dim,self._start_res,self._output_res)) #makes use of __call__ defined below
                sample_directory = image_sample_output + "epoch_{}".format(self._epochs_trained)
                GANutils.make_fresh_folder(sample_directory)
                for s,sample in enumerate(samples):
                    GANutils.create_image(GANutils.denormalise(sample), sample_directory+"/{}".format(s+1))

            if model_output:
                self.save_model(model_output)
            

    def __call__(self, latent_vector: np.array, noise_inputs: list[np.array]) -> np.array:
        pass

    def save_model(self, folder: str) -> None:
        pass

    def load_model():
        pass