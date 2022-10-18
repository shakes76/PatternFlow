import tensorflow as tf

from modules import *
from dataset import *

"""
Containing the source code for training, validating, testing and saving your model. 
The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”
Make sure to plot the losses and metrics during training.

"""

@tf.function
def train_step(siamese, siamese_optimiser, images1, images2, same_class: bool):
    """
    Executes one step of training the siamese model.
    Backpropogates to update weightings.

    Parameters:
        - siamese -- the siamese network
        - siamese_optimiser -- the optimiser which will be used for backprop
        - images1, images2 -- batch of image data which is either positive or negative
        - same_class -- flag representing whether the two sets of images are of the same class

    Returns:
        - loss value from this the training step
    
    """
    with tf.GradientTape() as siamese_tape:

        x0 = siamese(images1, training=True)
        x1 = siamese(images2, training=True)
        y = int(same_class)

        loss = siamese_loss(x0, x1, y)
    
        siamese_gradients = siamese_tape.gradient(\
            loss, siamese.trainable_variables)

        siamese_optimiser.apply_gradients(zip(
            siamese_gradients, siamese.trainable_variables))

        return loss