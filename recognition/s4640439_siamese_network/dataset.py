import numpy as np
from PIL import Image
import os
import time

# Data has already been separated into training and test data
AD_TEST_PATH = "E:/ADNI/AD_NC/test/AD/"
AD_TRAIN_PATH = "E:/ADNI/AD_NC/train/AD/"
NC_TEST_PATH = "E:/ADNI/AD_NC/test/NC/"
NC_TRAIN_PATH = "E:/ADNI/AD_NC/train/NC/"

# image constants
WIDTH = 256
HEIGHT = 240
CHANNELS = 1

PRE_PROC_DATA_SAVE_LOC = "E:/ADNI/Processed"

def load_data(directory_path, prefix):
  save_path = os.path.join(PRE_PROC_DATA_SAVE_LOC, f"{prefix}_preprocessed.npy")

  if not os.path.isfile(save_path):
    start = time.time()
    print("Processing data for file", save_path)

    data = []

    for filename in os.listdir(directory_path):
      path = os.path.join(directory_path, filename)

      img = Image.open(path)
      img_arr = np.asarray(img).astype(np.float32)

      # normalise
      img_arr = img_arr / 127.5 - 1
      data.append(img_arr)

    data = np.reshape(data, (-1, HEIGHT, WIDTH, CHANNELS))

    print("Saving data")
    np.save(save_path, data)

    elapsed = time.time() - start
    print (f'Image preprocess time: {elapsed}')

  else:
    print("Loading preprocessed data")
    data = np.load(save_path)

  return data

#training_data_positive = load_data(AD_TRAIN_PATH, "ad_train")
#training_data_negative = load_data(NC_TRAIN_PATH, "nc_train")
#testing_data_positive = load_data(AD_TEST_PATH, "ad_test")
#testing_data_negative = load_data(NC_TEST_PATH, "nc_test")