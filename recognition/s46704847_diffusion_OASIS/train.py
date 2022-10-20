"""
Reference: https://medium.com/@vedantjumle/image-generation-with-diffusion-
            models-using-keras-and-tensorflow-9f60aae72ac
"""

__author__ = "Zhao Wang, 46704847"
__email__ = "s4670484@student.uq.edu.au"

import numpy as np
from modules import Unet, generate_timestamp, forward_noise, get_checkpoint
from modules import loss_fn, show_forward_noise
from dataset import get_zipped_dataset, normalize, set_train_batch
import tensorflow as tf

# Parameters
PATH = "./test_dataset/test_dataset.zip" # The path of zipped image dataset
NUMBER_OF_SAMPLES = 64 # The number of image samples used to training
IMAGE_SIZE = (64, 64)
CKPT_PATH = "./checkpoint" # The path of the checkpoint
EPOCS = 50
BATCH_SIZE = 64 # set 64 in (64, 64) image size, 16 in (128, 128) image size , 
                # and 4 in (256, 256) image. otherwise, the graphic memory
                # will be "Out of Memory" in google Colab pro.

# Loading dataset
images = normalize(get_zipped_dataset(PATH, IMAGE_SIZE)[:NUMBER_OF_SAMPLES])
train_images = set_train_batch(images, BATCH_SIZE)

# Showing forward process
show_forward_noise(images)

# Creating Unet instance and checkpoint
unet, ckpt_manager = get_checkpoint(CKPT_PATH)

# Training model
rng = 0
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
def train_step(batch):
    rng, tsrng = np.random.randint(0, 300, size=(2,))
    timestep_values = generate_timestamp(tsrng, batch.shape[0])

    noised_image, noise = forward_noise(rng, batch, timestep_values)
    with tf.GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)
        
        loss_value = loss_fn(noise, prediction)
    
    gradients = tape.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return loss_value

for e in range(1, EPOCS+1):
    bar = tf.keras.utils.Progbar(len(train_images)-1)
    losses = []
    for i, batch in enumerate(iter(train_images)):
        # run the training loop
        loss = train_step(batch)
        losses.append(loss)
        bar.update(i, values=[("loss", loss)])

    avg = np.mean(losses)
    print(f"Average loss for epoch {e}/{EPOCS}: {avg}")
    ckpt_manager.save(checkpoint_number=e)