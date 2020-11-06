import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


input_path = 'input'

save_path = 'input/save-file.npy'

print(f"Looking for file: {save_path}")

if not os.path.isfile(save_path):
    print("Loading training images...")

    training_data = []
    for filename in tqdm(os.listdir(input_path)):
        path = os.path.join(input_path, filename)
        image = Image.open(path)
        training_data.append(np.asarray(image))
    training_data = np.reshape(training_data, (-1, 256, 256, 1))
    training_data = training_data.astype(np.float32)
    training_data = training_data / 127.5 - 1.

    print("Saving training image binary...")
    np.save(save_path, training_data)
else:
    print("Loading previous training pickle...")
    training_data = np.load(save_path)

BUFFER_SIZE = 60000
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)