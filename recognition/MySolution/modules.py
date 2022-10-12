import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import dataset


def main():
    dataset.handle_dataset()
    print("ok!")


if __name__ == "__main__":
    main()
