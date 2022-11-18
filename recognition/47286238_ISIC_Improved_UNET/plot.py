import torch
from matplotlib import pyplot as plt
import pickle

"""
Helper module to plot training metrics
"""

if __name__ == '__main__':
    with open('train_dsc.pkl', 'rb') as f:
        train_dsc = pickle.load(f)
    with open('train_loss.pkl', 'rb') as f:
        train_loss = pickle.load(f)
    with open('validation_dsc.pkl', 'rb') as f:
        validation_dsc = pickle.load(f)
    with open('validation_loss.pkl', 'rb') as f:
        validation_loss = pickle.load(f)
    
    # move tensors to cpu
    train_dsc = [x.cpu() for x in train_dsc]
    train_loss = [x.item() for x in train_loss]
    validation_dsc = [x.cpu() for x in validation_dsc]
    validation_loss = [x.item() for x in validation_loss]

    fig = plt.figure()

    # plot DSC change over epochs
    plt.title('Change in Dice Similarity Coefficient over Epochs')
    plt.plot(range(len(train_dsc)), train_dsc, label='Training DSC')
    plt.plot(range(len(train_dsc)), validation_dsc, label='Validation DSC')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Similarity Coefficient')
    plt.legend()
    plt.savefig('static/dsc_history.png')

    plt.clf()
    # plot Loss change pover opeepochs
    plt.title('Change in DSC Loss over Epochs')
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    plt.plot(range(len(validation_loss)), validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('DSC Loss')
    plt.legend()
    plt.savefig('static/loss_history.png')
    # plt.show()
