import tensorflow as tf
import tensorflow_addons as tfa



class IUNET(tf.keras.Model):
    """
        Improved UNet model
        
        The architecture for the improved UNet is based on the paper:
            "Brain Tumor Segmentation and Radiomics Survival Prediction:
             Contribution to the BRATS 2017 Challenge" [1]
        Available from: [https://arxiv.org/pdf/1802.10508v1.pdf] 
    
    """

    def __init__(self):
        super(IUNET, self).__init__()
        self.padding = "same"
        self.initial_output = 16
        self.contextDropoutRate = 0.3
        self.leakyAlpha = 1e-2
        
    
    def contextModule(self, input, outputFilters):
        """
            Defines a context module in the system. From [1]:
                "Each context module is in fact a pre-activation 
                 residual block with two 3x3 convolutional layers
                 and a dropout layer (0.3) in between."
                 
                 
            Pre-activation residual block sourced from: 
                1: "Identity Mappings in Deep Residual Networks"
            Available from: [https://arxiv.org/pdf/1603.05027.pdf]
                2: ResearchGate
            [https://www.researchgate.net/figure/Architecture-of-normal-residual-block-a-and-pre-activation-residual-block-b_fig2_337691625]
            
            - Batch normalisation was replaced with instance 
              normalisation, as mentioned in [1].
            - ReLU activations were replaced with Leaky ReLU
              activations, as mentioned in [1].
              
            @param input: the input to the context module
            @param outputFilters: the number of filters that this 
                                  particular context module will
                                  output

            @return the resultant input after being transformed by 
                    the context module.
        """
        
        batchOutput = tfa.layers.InstanceNormalization()(input)
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(batchOutput)
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(3,3), padding=self.padding)(reluActivation)
        
        afterDropout = tf.keras.layers.Dropout(self.contextDropoutRate)(convolutionOutput)
        
        batchOutput = tfa.layers.InstanceNormalization()(afterDropout)
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(batchOutput)
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(3,3), padding=self.padding)(reluActivation)
        
        return convolutionOutput
        
    
    def summation(self, fromConvolution, fromContextModule):
        """
            Performs an element-wise summation of the two inputs passed
            in.
            
            @param fromConvolution: the first input (usually from a 
                                    convolutional block)
            @param fromContextModule: the second input (usually 
                                      from a context module)
                                      
            @return the element-wise summation of the inputs
        """
        
        addOutput = tf.keras.layers.Add()([fromConvolution, fromContextModule])
        
        return addOutput
    
    
    def performUpSampling(self, input, outputFilters):
        """
            Defines an Up-Sampling module in the system. From [1]:
                "... This is achieved by first upsampling ... 
                 by means of a simple upscale that repeats the 
                 feature voxels twice in each spatial dimension, 
                 followed by a 3x3 convolution..."
                 
            - A leaky ReLU activation was added after the convolution 
            - An Instance Normalisation was added after the activation 
            
            @param input: the input to the up-sampling module 
            @param outputFilters: the amount of filters the upsampling 
                                  module will output.
                                  
            @return the resultant input after being transformed by the 
                    up-sampling module.
        """
        
        # Upscale, repeating the feature voxels twice in each dimension
        upSample = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(input)
        
        # 3x3 Convolution 
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(3,3), padding=self.padding)(upSample)
        
        # Leaky ReLU activation 
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        
        # Perform normalisation 
        reluActivation = tfa.layers.InstanceNormalization()(reluActivation)
        
        return reluActivation
        
    
    
    def performConcatenation(self, fromLower, skipConnection):
        """
            Performs a concatenation of the two inputs passed.
            
            @param fromLower: the first input (usually from an up-sampling 
                              module)
            @param skipConnection: the second input (usually a skip 
                                   connection from earlier in the network)
                                   
            @return the concatenation of the two input layers
        """
        
        concatenation = tf.keras.layers.Concatenate()([fromLower, skipConnection])
        
        return concatenation

    
    def localisationModule(self, input, outputFilters):
        """
            Defines a Localisation module in the system. From [1]:
                " A localisation module consists of a 3x3 
                  convolution followed by a 1x1 convolution 
                  ... "
                  
            - A leaky ReLU activation was added after the convolutions 
            - An Instance Normalisation was added after the activation 
            
            @param input: the input to the localisation module 
            @param outputFilters: the amount of filters the 
                                  localisation module will output
                                  
            @return the resultant input after being transformed by the 
                    localisation module
        """
        
        # 3x3 Convolution 
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(3,3), padding=self.padding)(input)
        
        # Leaky ReLU activation 
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        
        # Perform normalisation 
        reluActivation = tfa.layers.InstanceNormalization()(reluActivation)
        
        # 1x1 Convolution
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(1,1), padding=self.padding)(reluActivation)
        
        # Leaky ReLU activation 
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        
        # Perform normalisation 
        reluActivation = tfa.layers.InstanceNormalization()(reluActivation)
        
        return reluActivation
    
    
    def performSegmentation(self, input, outputFilters):
        """
            Performs segmentation on the input. A segmentation layer is 
            often a 1x1 convolution with a single output filter.
            
            @param input: the input to be segmented 
            @param outputFilters: the amount of filters to be output 
                                  (always 1)
                                  
            @return the segmented input
        """
        # 1x1 Convolution 
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(1,1), padding=self.padding)(input)
        
        # Leaky ReLU activation 
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        
        return reluActivation
    
    def createPipeline(self):
        """
            Creates the pipeline for the Improved UNet Architecture [1].
        """
        print("TFA version: " + tfa.__version__)
        
        input = tf.keras.layers.Input(shape=(192, 256, 3))
        
        ## Encoder 
        # Encoder, level one.
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output, kernel_size=(3,3), padding=self.padding)(input)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        contextOutput = self.contextModule(convolutionOutput, self.initial_output)
        sumOfOutputs = self.summation(convolutionOutput, contextOutput)
        firstSkip = sumOfOutputs
        
        # Encoder, level two.
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output * 2, kernel_size=(3,3), padding=self.padding, strides=(2,2))(sumOfOutputs)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        contextOutput = self.contextModule(convolutionOutput, self.initial_output * 2)
        sumOfOutputs = self.summation(convolutionOutput, contextOutput)
        secondSkip = sumOfOutputs
        
        # Encoder, level three.
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output * 4, kernel_size=(3,3), padding=self.padding, strides=(2,2))(sumOfOutputs)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        contextOutput = self.contextModule(convolutionOutput, self.initial_output * 4)
        sumOfOutputs = self.summation(convolutionOutput, contextOutput)
        thirdSkip = sumOfOutputs
        
        # Encoder, level four.
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output * 8, kernel_size=(3,3), padding=self.padding, strides=(2,2))(sumOfOutputs)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        contextOutput = self.contextModule(convolutionOutput, self.initial_output * 8)
        sumOfOutputs = self.summation(convolutionOutput, contextOutput)
        fourthSkip = sumOfOutputs
        
        ## Level 5: Bottom of network.
        # Convolutions / Context modules as before
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output * 16, kernel_size=(3,3), padding=self.padding, strides=(2,2))(sumOfOutputs)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        contextOutput = self.contextModule(convolutionOutput, self.initial_output * 16)
        sumOfOutputs = self.summation(convolutionOutput, contextOutput)
        
        # Perform upsampling
        upSampleOutput = self.performUpSampling(sumOfOutputs, self.initial_output * 8)
        
        # Concatenate
        concatenated = self.performConcatenation(upSampleOutput, fourthSkip)
  
        ### Decoder
        ## Decoder, level four.
        localisationOutput = self.localisationModule(concatenated, self.initial_output * 8)
        upSampleOutput = self.performUpSampling(localisationOutput, self.initial_output * 4)
        
        ## Decoder, level three.
        concatenated = self.performConcatenation(upSampleOutput, thirdSkip)
        localisationOutput = self.localisationModule(concatenated, self.initial_output * 4)
        toSegmentLower = localisationOutput
        
        # Perform first segmentation
        lowerSegmented = self.performSegmentation(toSegmentLower, 1) 
        
        # Upsample as usual
        upSampleOutput = self.performUpSampling(localisationOutput, self.initial_output * 2)
        
        ## Decoder, level two.
        concatenated = self.performConcatenation(upSampleOutput, secondSkip)
        localisationOutput = self.localisationModule(concatenated, self.initial_output * 2)
        toSegmentMiddle = localisationOutput
        
        # Perform second segmentation
        middleSegmented = self.performSegmentation(toSegmentMiddle, 1) 
        
        # Upsample as usual 
        upSampleOutput = self.performUpSampling(localisationOutput, self.initial_output)
        
        ## First Skip-Add 
        # Add together the middleSegmented and lowerSegmented
        # lowerSegmented must be up-scaled first.
        upScaledLowerSegment = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(lowerSegmented)
        
        # Element-wise sum
        firstSkipSum = self.summation(upScaledLowerSegment, middleSegmented)
        
        ## Decoder, level one.
        concatenated = self.performConcatenation(upSampleOutput, firstSkip)
        
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output * 2, kernel_size=(3,3), padding=self.padding)(concatenated)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        
        # Perform segmentation 
        upperSegmented = self.performSegmentation(convolutionOutput, 1)
        
        ## Second Skip-Add
        # Add together the middleSegmented and upperSegmented 
        # middleSegmented must be up-scaled first
        upScaledMiddleSegment = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(firstSkipSum)
        finalNode = self.summation(upScaledMiddleSegment, upperSegmented)
        
        # Final network activation 
        networkActivation = tf.keras.layers.Activation('sigmoid')(finalNode)
        
        model = tf.keras.Model(inputs=input, outputs=networkActivation)
        #model.summary()
        return model