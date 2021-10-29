import pandas as pd
import numpy as np

def load_data():
    """ loads the preprocessed data from provided facebook.npz file
    """
    data = np.load("facebook.npz")
