# This file contains the source code of the components of my model as functions or classes
import os
import sys
sys.path.insert(1, os.getcwd())
from dataset import loadFile

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

