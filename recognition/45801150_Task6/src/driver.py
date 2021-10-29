import numpy as np
import VQVAE
import load_oasis_data

x_train, x_test, x_val = load_oasis_data.get_data()

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_val = np.expand_dims(x_val, -1)

x_train_normalised = (x_train / 255.0) - 0.5
x_test_normalised = (x_test / 255.0) - 0.5
x_val_normalised = (x_val / 255.0) - 0.5

variance = np.var(x_train / 255.0)

vqvae = VQVAE.train_vqvae(x_train_normalised, variance, x_val_normalised)
VQVAE.compare_reconstructions(vqvae, x_test_normalised, 10)

