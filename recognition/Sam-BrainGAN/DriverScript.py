import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from GANModule import run_GAN


input_path = 'input'

save_path = 'input/save-file.npy'

print(f"Looking for file: {save_path}")

# Creates a save file of the brain dataset if it doesn't already exist.
# Else loads it to greatly reduce startup cost.
if not os.path.isfile(save_path):
    print("Loading training images...")

    training_data = []
    for filename in tqdm(os.listdir(input_path)):
        path = os.path.join(input_path, filename)
        image = Image.open(path)
        training_data.append(np.asarray(image))
    
    # Reshapes and normalises the data
    training_data = np.reshape(training_data, (-1, 256, 256, 1))
    training_data = training_data.astype(np.float32)
    training_data = training_data / 127.5 - 1.

    print("Saving training image binary...")
    np.save(save_path, training_data)
else:
    print("Loading previous training pickle...")
    training_data = np.load(save_path)

BUFFER_SIZE = 60000
BATCH_SIZE = 4

# Converts the numpy array to a BatchDataset tensor.
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Runs the GAN module
run_GAN(train_dataset)