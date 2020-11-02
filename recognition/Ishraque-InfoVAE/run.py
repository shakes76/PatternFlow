import tensorflow as tf
from data import train_x, test_x, valid_x

# normalise data
train_x = train_x.map(lambda x: x / 255)
test_x = test_x.map(lambda x: x / 255)
valid_x = valid_x.map(lambda x: x / 255)

