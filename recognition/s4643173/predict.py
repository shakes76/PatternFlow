from dataset import *
from modules import *
from train import *

import os
import pickle

def main():
    if 'VQVAE' not in os.listdir('.'):
        print('No saved VQVAE model found. Creating and training a new model.')
        print('-' * 50)
        fit_vqvae()
    if 'Gen' not in os.listdir('.'):
        print('No saved DCGAN model found. Creating and training a new model.')
        print('-' * 50)
        fit_gan()

    with open('VQVAE', 'rb') as f:
        vae_model = pickle.load(f)
    with open('Gen', 'rb') as f:
        gen = pickle.load(f)

if __name__ == '__main__':
    main()

