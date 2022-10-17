import os
from PIL import Image
import glob
import numpy as np

class DataLoader():
    def __init__(self):
        super()
        # self.train_paths = train_paths
        # self.test_paths = test_paths


    def fetch_data(self, paths):
        ds = []
        for p in paths:
            for f in glob.iglob(p+'/*'):
                ds.append(np.asarray(Image.open(f)))
        
        ds = np.array(ds)
        ds = np.expand_dims(ds, -1)
        return ds

    def preprocessing(self, dataset):
        dataset_normed = (dataset / 255.0) - 0.5
        return dataset_normed

    def get_variance(dataset):
        return np.var(dataset / 255.0)


# train_path = ["/content/drive/My Drive/comp3710/project/keras_png_slices_data/keras_png_slices_data/keras_png_slices_train", "/content/drive/My Drive/comp3710/project/keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_train"]
# test_path = ["/content/drive/My Drive/comp3710/project/keras_png_slices_data/keras_png_slices_data/keras_png_slices_test", "/content/drive/My Drive/comp3710/project/keras_png_slices_data/keras_png_slices_data/keras_png_slices_test"]
