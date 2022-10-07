import os
import tensorflow as tf
from tensorflow import keras

dataset_url =  "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"
data_dir = keras.utils.get_file(origin=dataset_url, fname="ADNI-MRI", extract=True)
root_dir = os.path.join(data_dir, "ADNI-MRI/data")

print(root_dir)
