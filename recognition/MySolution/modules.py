import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from train import *
from dataset import *


"""
Note: All the code used in this file has been inspired by
https://keras.io/examples/vision/image_classification_with_vision_transformer/
"""

"""
Vision transformer class that contains all the model data
"""
class VIT:
    """
    Constructor used to call main
    """
    def __init__(self):
        self.main()

    def main(self):
        LoadData() # Loads all the given data

        x_train = np.load('X_train.npy') # Loads x_train data

        # All the parameters for the Vision Transformer
        num_classes = 1 # AD or NC
        input_shape = (240, 256, 1)
        image_size = 250
        patch_size = 32
        num_patches = (image_size // patch_size) ** 2
        projection_dim = 64 
        num_heads = 8 
        transformer_units = [
            projection_dim * 3,
            projection_dim,
        ] 
        # size of the transformer layers
        transformer_layers = 9
        # Size of the dense layers of the final classifier
        mlp_head_units = [2048, 256]

        #Plots all the images
        plt.figure(figsize=(4, 4))
        image = x_train[np.random.choice(range(x_train.shape[0]))]
        plt.imshow(image.astype("uint8"), cmap="gray")
        plt.axis("off")

        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(image_size, image_size)
        )
        plt.savefig("OG.jpg") 

        # Prints out all the data for the images
        patches = Patches(patch_size)(resized_image)    
        print(f"Image size: {image_size} X {image_size}", flush=True)
        print(f"Patch size: {patch_size} X {patch_size}", flush=True)
        print(f"Patches per image: {patches.shape[1]}", flush=True)
        print(f"Elements per patch: {patches.shape[-1]}", flush=True)

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))

        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (patch_size, patch_size, 1))
            plt.imshow(patch_img.numpy().astype("uint8"), cmap="gray")
            plt.axis("off")

        plt.savefig("Test.jpg")

        # Data Augmentation used to improve model
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.Normalization(),
                layers.experimental.preprocessing.Resizing(image_size, image_size),
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(factor=0.2),
                layers.experimental.preprocessing.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation"
        )   
        
        data_augmentation.layers[0].adapt(x_train)
        inputs = layers.Input(shape=input_shape)
        augmented = data_augmentation(inputs)
        patches = Patches(patch_size)(augmented) # Creates patches
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches) # Applies Patch Encoder

        # Loops through all transformer layers
        for _ in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        # Adds additional layers
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        
        # Calls the Multi layer Perceptron
        features = self.mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Final layer of the model. Uses Sigmoid activation as it works the best (based on trial and error)
        logits = layers.Dense(num_classes, activation="sigmoid")(features)

        model = keras.Model(inputs=inputs, outputs=logits)
        model.summary() # Outputs a model Summary
        Train(model=model) # Trains the model

    """
    Function for the multi layer perceptron.
    """
    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x


"""
Patches class used for creating patches
"""
class Patches(layers.Layer):

    """
    Constructor for the Patch class
    """
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    """
    Creates the patches and returns it
    """
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1]  ,
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""
Patch Encoding class used to encode Patches
"""
class PatchEncoder(layers.Layer):

    """
    Constructor for the Patch encoder class
    """
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    """
    Returns encoded Patch
    """
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

VIT() # Calls the Vision Transformer class to begin running the model