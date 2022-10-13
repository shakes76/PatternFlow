import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
# from tensorflow.keras import layers, losses, models, regularizers

class DataLoader:
    def __init__(self):
        pass
    
    def parse_data(self):
        data_url = "https://graphmining.ai/datasets/ptg/facebook.npz"
        data_path = 'facebook.npz'

        if not os.path.exists(data_path):
            print(f'Downloading from {data_url}, this may take a while...')
            utils.get_file('facebook.npz', data_url)

        with np.load(data_path) as data:
            edges = data['edges']
            features = data['features']
            target = data['target']

        return  
            