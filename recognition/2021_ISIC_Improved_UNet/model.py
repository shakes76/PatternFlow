import tensorflow as tf

# Initial model file

class IUNET(tf.keras.Model):
    """Improved UNet model"""

    def __init__(self):
        super(IUNET, self).__init__()
        self.padding = "same"
        
    
    def contextModule(self, input):
        print("Defines the architecture of a Context Module")
        
    
    def createPipeline(self):
        print("Creates the model architecture")
        
        input = tf.keras.layers.Input(shape=(256,256,3))
        
        ## Encoder 
        # Encoder, level one.
        convolutionOutput = tf.keras.layers.Conv2D(16, kernel_size=(3,3), padding=self.padding)(input)
        contextOutput = self.contextModule(convolutionOutput)
        
        ## Decoder
        
        
        #model = 
        # return model
        return 1