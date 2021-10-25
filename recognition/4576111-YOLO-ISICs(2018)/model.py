import tensorflow as tf

class YOLOV1():

    def __init__(self, S=7, B=2, C=1):
        self.S = S
        self.B = B
        self.C = C
        self.model = self.modelDefinition()

    

    def modelDefinition(self):
        """Defines the YoloV1 neural network. 
        Batch Normalization has been introduced to make the network more stable and increase peformance.
        A sigmoid activation function has been introduced to speed up training time (over linear activation). 

        Returns:
            tensorflow.keras.Sequential: A Convolutional Neural Network defined by the YoloV1 architecture.  
        """
        model = tf.keras.Sequential([
            #First Layer
            tf.keras.layers.Conv2D(64, (7,7), strides=(2, 2),  input_shape=(488,488,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
            
            #Second Layer
            tf.keras.layers.Conv2D(192, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
            
            #Third Layer    
            tf.keras.layers.Conv2D(128, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
            
            #Fourth Layer
            # +++ Repeated block
            tf.keras.layers.Conv2D(256, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(256, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(512, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            # +++ END BLOCK
            tf.keras.layers.Conv2D(512, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
            
            #Fifth layer
            # +++ Repeated Block
            tf.keras.layers.Conv2D(512, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(512, (1,1),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            # +++ END BLOCK
            tf.keras.layers.Conv2D(1024, (3,3),  strides=(2, 2), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            #Sixth Layer
            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            tf.keras.layers.Conv2D(1024, (3,3),  padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            
            # Final Output Layer
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096),
            tf.keras.layers.Dense(self.S * self.S * (self.B*5+self.C), input_shape=(4096,), activation="sigmoid"),
            tf.keras.layers.Reshape(target_shape = (self.S, self.S, (self.B*5+self.C)))
        ])

        return model

