import tensorflow as tf

#Create subclasses for layers and model
class ConvBlock(tf.keras.layers.Layer):
    """
    Subclass for a convolution block (2 convolution layers, 1 maxpooling layer and 1 dropout layer)
    """
    def __init__(self, filters=32):
        super(ConvBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.drop = tf.keras.layers.Dropout(0.2)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return self.drop(x)

class CNNModel(tf.keras.Model):
    """
    Subclass for the CNN model. 3 convolution blocks, 1 flatten layer, 2 dense layers and 1 dropout layer.
    """
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        self.block1 = ConvBlock(32)
        self.block2 = ConvBlock(64)
        self.block3 = ConvBlock(128)
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.5)
        self.d2sm = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.drop(x)
        return self.d2sm(x)