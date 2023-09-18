import os
from PIL import Image
import glob
import numpy as np

'''
Class contains all functionalities to load and preprocess the dataset.
'''
class DataLoader():
    def __init__(self):
        super()
        # self.train_paths = train_paths
        # self.test_paths = test_paths


    '''
    Take a list of paths and search all the images under the paths.
    Returns a list of images as numpy array.
    '''
    def fetch_data(self, paths):
        ds = []
        for p in paths:
            for f in glob.iglob(p+'/*'):
                ds.append(np.asarray(Image.open(f)))
        
        ds = np.array(ds)
        ds = np.expand_dims(ds, -1)
        return ds

    '''
    Normalize and scale dataset.
    '''
    def preprocessing(self, dataset):
        dataset_normed = (dataset / 255.0) - 0.5
        return dataset_normed

    '''
    Returns the variance of the dataset.
    '''
    def get_variance(dataset):
        return np.var(dataset / 255.0)



