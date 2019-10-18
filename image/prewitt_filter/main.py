import glob,os
import pathlib
import tensorflow as tf

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from prewitt import prewitt_filter
tf.enable_eager_execution()
data_dir = pathlib.Path('./resources/')
image_paths = list(data_dir.glob('./*'))

dataset = tf.data.Dataset.list_files(str(data_dir/'*'))


