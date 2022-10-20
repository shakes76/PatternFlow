import numpy as np 
import os
from PIL import Image

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

X_training = []
X_train_labels = []
for fname in os.listdir(os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/train/AD")):
    fpath = os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/train/AD", fname)
    im = Image.open(fpath)
    X_training.append(np.array(im))
    X_train_labels.append(1)
    im.close()

for fname in os.listdir(os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/train/NC")):
    fpath = os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/train/NC", fname)
    im = Image.open(fpath)
    X_training.append(np.array(im))
    X_train_labels.append(0)
    im.close()

# Convert to nparray
X_training = np.array(X_training)
X_train_labels = np.array(X_train_labels)
print(X_training.shape)

x_test = []
y_test = []
for fname in os.listdir(os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/test/AD")):
    fpath = os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/test/AD", fname)
    im = Image.open(fpath)
    x_test.append(np.array(im))
    y_test.append(1)
    im.close()

for fname in os.listdir(os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/test/NC")):
    fpath = os.path.join(__location__, "ADNI_AD_NC_2D/AD_NC/test/NC", fname)
    im = Image.open(fpath)
    x_test.append(np.array(im))
    y_test.append(0)
    im.close()

# Convert to nparray
x_test = np.array(x_test)
y_test = np.array(y_test)
print(max(x_test[0][120]))