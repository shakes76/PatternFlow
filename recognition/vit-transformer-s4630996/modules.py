"""
Assumptions:

Steps / Key Functions:
1. Augment Data 
2. Create Patches
3. Embed Patches
4. Create MLP 
5. Create Transformer Encoder
5. Create ViT

References:
1) https://keras.io/examples/vision/image_classification_with_vision_transformer/
2) https://towardsdatascience.com/understand-and-implement-vision-transformer-with-tensorflow-2-0-f5435769093

"""

##############################  INPUT DATA AUGMENTATION  ###################################

def data_augmentation(mean, variance):
    """ data augmentation for input data based on calculated mean and variance of training data """

    data_augmentation = keras.Sequential(
        [
            layers.Normalization(mean=mean, variance=variance),
        ],
        name="data_augmentation",
    )

    return data_augmentation