import tensorflow as tf
import tensorflow_addons as tfa

# Initial model file

## References
# residual block / context module architecture: https://www.researchgate.net/figure/Architecture-of-normal-residual-block-a-and-pre-activation-residual-block-b_fig2_337691625
# residual block: https://arxiv.org/pdf/1603.05027.pdf
# batch normalisation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
# dropout layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
# leakyRelu activation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU
# sigmoid activation: https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid

class IUNET(tf.keras.Model):
    """Improved UNet model"""

    def __init__(self):
        super(IUNET, self).__init__()
        self.padding = "same"
        self.initial_output = 16
        self.contextDropoutRate = 0.3
        self.leakyAlpha = 1e-2
        
    
    def contextModule(self, input, outputFilters):
        print("Defines the architecture of a Context Module")
        
        batchOutput = tfa.layers.InstanceNormalization()(input)
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(batchOutput)
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(3,3), padding=self.padding)(reluActivation)
        
        afterDropout = tf.keras.layers.Dropout(self.contextDropoutRate)(convolutionOutput)
        
        batchOutput = tfa.layers.InstanceNormalization()(afterDropout)
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(batchOutput)
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(3,3), padding=self.padding)(reluActivation)
        
        return convolutionOutput
        
    
    def summation(self, fromConvolution, fromContextModule):
        print("Defines the summation of the convolution and context module outputs")
        addOutput = tf.keras.layers.Add()([fromConvolution, fromContextModule])
        return addOutput
    
    
    def performUpSampling(self, input, outputFilters):
        print("Defines the behaviour of the Up Sampling module")
        
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
        print("Defines the concatenation of the data from the lower parts of the network and the skip connection")
        concatenation = tf.keras.layers.Concatenate()([fromLower, skipConnection])
        return concatenation

    
    def localisationModule(self, input, outputFilters):
        print("Defines the localisation module")
        
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
        print("Defines the segmentation layer")
        # 1x1 Convolution 
        convolutionOutput = tf.keras.layers.Conv2D(outputFilters, kernel_size=(1,1), padding=self.padding)(input)
        
        # Leaky ReLU activation 
        reluActivation = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        
        return reluActivation
    
    def createPipeline(self):
        print("Creates the model architecture")
        
        input = tf.keras.layers.Input(shape=(192, 256, 3))
        
        ## Encoder 
        # Encoder, level one.
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output, kernel_size=(3,3), padding=self.padding)(input)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        convolutionOutput = tfa.layers.InstanceNormalization()(convolutionOutput)
        #convolutionOutput = tf.keras.layers.Dropout(0.1)(convolutionOutput)
        contextOutput = self.contextModule(convolutionOutput, self.initial_output)
        sumOfOutputs = self.summation(convolutionOutput, contextOutput)
        firstSkip = sumOfOutputs
        
        # Encoder, level two.
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output * 2, kernel_size=(3,3), padding=self.padding, strides=(2,2))(sumOfOutputs)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        convolutionOutput = tfa.layers.InstanceNormalization()(convolutionOutput)
        #convolutionOutput = tf.keras.layers.Dropout(0.1)(convolutionOutput)
        contextOutput = self.contextModule(convolutionOutput, self.initial_output * 2)
        sumOfOutputs = self.summation(convolutionOutput, contextOutput)
        secondSkip = sumOfOutputs
        
        # Encoder, level three.
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output * 4, kernel_size=(3,3), padding=self.padding, strides=(2,2))(sumOfOutputs)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        convolutionOutput = tfa.layers.InstanceNormalization()(convolutionOutput)
        #convolutionOutput = tf.keras.layers.Dropout(0.1)(convolutionOutput)
        contextOutput = self.contextModule(convolutionOutput, self.initial_output * 4)
        sumOfOutputs = self.summation(convolutionOutput, contextOutput)
        thirdSkip = sumOfOutputs
        
        # Encoder, level four.
        convolutionOutput = tf.keras.layers.Conv2D(self.initial_output * 8, kernel_size=(3,3), padding=self.padding, strides=(2,2))(sumOfOutputs)
        convolutionOutput = tf.keras.layers.LeakyReLU(alpha=self.leakyAlpha)(convolutionOutput)
        convolutionOutput = tfa.layers.InstanceNormalization()(convolutionOutput)
        #convolutionOutput = tf.keras.layers.Dropout(0.1)(convolutionOutput)
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
        #networkActivation = tf.keras.activations.sigmoid(finalNode)
        networkActivation = tf.keras.layers.Activation('sigmoid')(finalNode)
        
        model = tf.keras.Model(inputs=input, outputs=networkActivation)
        #model.summary()
        return model