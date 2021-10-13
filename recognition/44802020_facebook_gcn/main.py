from typing import Any
import model
import tensorflow as tf
import numpy as np


# Constants
def summarise_data(data, aspect):
    print(f"===== {aspect} =====")
    aspect_d = data[aspect]
    print(aspect_d.shape)  # (22 470, 128)
    print(aspect_d)
    print(type(aspect_d))
    print(type(aspect_d[0]))
    print(aspect_d[0])
    print("====================")


def main():
    print("Tensorflow version:", tf.__version__)
    print("Numpy version:", np.__version__)

    # file_path = r"C:\Users\johnv\Documents\Code Projects\Pattern Recognition\facebook.npz"

    file_path = r"C:\Users\johnv\Documents\University\COMP3710\Pattern Flow Project\facebook.npz"

    # Load in Data
    data = np.load(file_path)

    # print(data.files)

    features = data['features']
    edges = data['edges']
    target = data['target']

    # summarise_data(data, 'features')
    # summarise_data(data, 'edges')
    # summarise_data(data, 'target')

    # There are 22 470 Pages
    # Each with 128 features
    # Each falls into 1 of 4 categories
    # There are 342 004 Edges between the pages

    model.Model(data)


if __name__ == '__main__':
    main()



