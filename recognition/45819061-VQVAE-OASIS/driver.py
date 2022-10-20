from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import BATCH_SIZE, get_data
from modules import VQVAE, get_pixelcnn, get_pixelcnn_sampler
from train import *

x_train, x_test, x_validate = get_data()
model = train(x_train, x_test, x_validate)
demo_model(model,  x_test)
pixelcnn = pixelcnn_train(model, x_train, x_test, x_validate)