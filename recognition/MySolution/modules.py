import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import dataset
import train


def main():
    dataset.handle_dataset()
    # train.handle_training(sorted_data)
    print("ok!")


if __name__ == "__main__":
    main()
