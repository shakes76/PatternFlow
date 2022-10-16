# This file contains the data loader
from cgitb import grey
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def loadFile(dir):
    print('>> Begin data loading')
    path = {'train_ad': os.path.join(dir, 'train/AD'),
            'train_nc': os.path.join(dir, 'train/NC'),
            'test_ad': os.path.join(dir, 'test/AD'),
            'test_nc': os.path.join(dir, 'test/NC')}
    
    print('-Directory of the Training files of class AD is: {}'.format(path['train_ad']))
    print('-Directory of the Training files of class NC is: {}'.format(path['train_nc']))
    print('-Directory of the Testing files of class AD is: {}'.format(path['test_ad']))
    print('-Directory of the Testing files of class NC is: {}'.format(path['test_nc']))
    # print('\n> 1/6 Loading Training Data of class AD...')
    
    ds = {'train_ad': [],
          'train_nc': [],
          'test_ad': [],
          'test_nc': [],
          'valid_ad': [],
          'valid_nc': []}
    # load images in train and test folders
    for which in path:
        print('> Loading image in {}'.format(which))
        for file in os.listdir(path[which]):
            filePath = os.path.join(path[which], file)
            # load and convert image to greyscale
            image = np.asarray(Image.open(filePath).convert('L'))
            if image is not None:
                ds[which].append(image)
    # Split training data to obtain validation data (30%)
    print('> Generate validation set')
    print('>> Shuffling ...')
    random.shuffle(ds['train_ad'])
    random.shuffle(ds['train_nc'])
    random.shuffle(ds['test_ad'])
    random.shuffle(ds['test_nc'])
    train_ad = ds['train_ad']
    train_nc = ds['train_nc']
    print('>> Extracting image to form validation set ...')
    valid_ad = train_ad[:round(len(train_ad)*0.3)]
    valid_nc = train_nc[:round(len(train_nc)*0.3)]
    train_ad = train_ad[round(len(train_ad)*0.3):]
    train_nc = train_nc[round(len(train_nc)*0.3):]
    ds['train_ad'] = train_ad
    ds['train_nc'] = train_nc
    ds['valid_ad'] = valid_ad
    ds['valid_nc'] = valid_nc
    print('> Completed')
            
    return ds['train_ad'], ds['train_nc'], ds['valid_ad'], ds['valid_nc'], ds['test_ad'], ds['test_nc']
    
def plotExample(ds):

    for x in ds:
        print(x.shape)
        print(x[100])
        # print(len(x))
        plt.axis("off")
        plt.imshow((x), cmap='gray', vmin=0, vmax=255)
        plt.show()
        break

def main():
    # Code for testing the functions
    ta, tn, va, vn, ta, tn = loadFile('F:/AI/COMP3710/data/AD_NC/')
    plotExample(ta)

if __name__ == "__main__":
    main()

