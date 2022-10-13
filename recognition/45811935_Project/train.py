"""
    Performs all training/validating/testing/saving of models and plotting of results (i.e.
    losses and metrics during training/validation).

    Author: Adrian Rahul Kamal Rajkamal
    Student Number: 45811935
"""
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from dataset import *
from modules import *