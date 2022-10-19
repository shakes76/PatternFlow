import tensorflow as tf

test_directory = 'test'
train_directory = 'train'

def load_test():
    data = tf.keras.utils.image_dataset_from_directory(test_directory,image_size=(256, 256),color_mode='grayscale', label_mode='categorical')
    norm = tf.keras.layers.Rescaling(1./255)
    norm_data = data.map(lambda x,y:(norm(x),y))
    return norm_data


def load_train_data():
    train_data = tf.keras.utils.image_dataset_from_directory(train_directory,image_size=(256, 256),color_mode='grayscale', label_mode='categorical',seed = 12, validation_split=0.25, subset='training')
    valid_data = tf.keras.utils.image_dataset_from_directory(train_directory,image_size=(256, 256),color_mode='grayscale', label_mode='categorical',seed = 12, validation_split=0.25, subset='validation')
    norm = tf.keras.layers.Rescaling(1./255)
    train_data = train_data.map(lambda x,y:(norm(x),y))
    valid_data = valid_data.map(lambda x,y:(norm(x),y))
    return train_data, valid_data







from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def torch_train(batch_size=64,validation_split=0.25):
    transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
    dataset = datasets.ImageFolder('train',transform=transform)
    size = len(dataset)
    valid_size = int(np.floor(validation_split*size))
    indicies = list(range(size))
    np.random.shuffle(indicies)
    train_indices, val_indices = indicies[valid_size:], indicies[:valid_size]
    tsampler = SubsetRandomSampler(train_indices)
    vsampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size,sampler=tsampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,sampler=vsampler)
    return train_loader, validation_loader

def torch_test(batch_size=64):
    transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
    dataset = datasets.ImageFolder('test',transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader