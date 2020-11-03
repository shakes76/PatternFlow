import tensorflow as tf
# from data import train_x, test_x, valid_x
from model import InfoVAE
# normalise data
# train_x = train_x.map(lambda x: x / 255)
# test_x = test_x.map(lambda x: x / 255)
# valid_x = valid_x.map(lambda x: x / 255)
latent_dim = 2
iv = InfoVAE(latent_dim)