import numpy as np
import tensorflow as tf

from google.colab import drive

"""Load facebook data

"""

def load_data(path='/content/gdrive', file='/MyDrive/Colab Notebooks/Lab 3/facebook.npz'):
  drive.mount('/content/gdrive', force_remount=True);
  data = np.load("{}{}".format(path, file))
  print(data)

load_data()