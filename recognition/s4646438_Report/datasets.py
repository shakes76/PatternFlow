import tensorflow as tf

import os

from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

from IPython.display import display


dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"

data_path = keras.utils.get_file(origin=dataset_url, fname="ADNI", extract=True)
data_path = data_path[:-4]
train_path = os.path.join(data_path, "AD_NC/train")
test_path = os.path.join(data_path, "AD_NC/test")

batch_size = 8
image_size = (500, 500)

train_ds = image_dataset_from_directory(
    train_path,
    batch_size=batch_size,
    image_size=image_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
    crop_to_aspect_ratio=True
)

validation_ds = image_dataset_from_directory(
    train_path,
    batch_size=batch_size,
    image_size=image_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
    crop_to_aspect_ratio=True
)

scale = lambda img : img / 255
# Scale from (0, 255) to (0, 1)
train_ds = train_ds.map(scale)
valid_ds = validation_ds.map(scale)

for batch in train_ds.take(1):
    for img in batch:
        display(array_to_img(img))
        print(np.max(img))
