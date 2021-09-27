from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math

''' LOAD IN DATA. Organize by patient, to prevent data leakage. '''

DIR = "AKOA_Analysis/"
file_paths = [DIR + x for x in os.listdir(DIR)]
new_patient_ids = {} # key: e.g. OAI9014797_BaseLine_3_L, value: new id
data = {} # key: unique patient id (created), value: ([xdata], [labels [0 for left, 1 for right]])

for file in file_paths:
    is_right = "RIGHT" in file
    patient_id = file.split("de3d1")[0].split("/")[1] + ("L" if not is_right else "R")
    if patient_id not in new_patient_ids:
        new_patient_ids[patient_id] = len(new_patient_ids)
    new_id = new_patient_ids[patient_id]
    img = np.asarray(Image.open(file).convert("L"))
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    label = 1 if is_right else 0
    if new_id in data:
        data[new_id][0].append(img)
        data[new_id][1].append(label)
    else:
        data[new_id] = ([img], [label])


''' SPLIT DATA. Get train/test split based on patients. '''

TEST_SPLIT = 0.4
num_patients = len(list(data.keys()))
patient_ids = list(range(0, num_patients))
test_patients = random.sample(patient_ids, int(num_patients*TEST_SPLIT))
train_patients = [x for x in patient_ids if x not in test_patients]

xtrain, xtest, ytrain, ytest = [], [], [], []
for pid in patient_ids:
    #print(data[pid])
    for idx in range(len(data[pid][0])):
        if pid in train_patients:
            xtrain.append(data[pid][0][idx])
            ytrain.append(data[pid][1][idx])
        else:
            xtest.append(data[pid][0][idx])
            ytest.append(data[pid][1][idx])
print(len(xtrain), len(xtest), len(ytrain), len(ytest))
del data


''' SHUFFLE DATA AND SAVE. '''

indices_train = list(range(0, len(xtrain)))
indices_test = list(range(0, len(xtest)))
random.shuffle(indices_train)
random.shuffle(indices_test)
xtrain = np.array(xtrain)
xtrain = xtrain[indices_train]
#np.save("xtrain", xtrain)
#del xtrain
xtest = np.array(xtest)
xtest = xtest[indices_test]
#np.save("xtest", xtest)
#del xtest
ytrain = np.array(ytrain)
ytrain = ytrain[indices_train]
#np.save("ytrain", ytrain)
#del ytrain
ytest = np.array(ytest)
ytest = ytest[indices_test]
#np.save("ytest", ytest)
del ytest

''' GET FOURIER FEATURES FOR POSITIONAL ENCODINGS. '''

def get_positional_encodings(img_data, bands=64, sampling_rate=10):
    # assume 2 dimensions, using single channel images
    #flattened = img_data.flatten()
    rows, cols = img_data.shape
    encodings = []
    xr = [2*(idx//cols)/(rows-1) - 1 for idx in list(range(rows*cols))] # rows in [-1,1] range
    xc = [2*(idx % cols)/(cols-1) - 1 for idx in list(range(rows*cols))] # cols in [-1,1] range
    for input in range(rows*cols):
        encoding = []
        for xd in [xr[input], xc[input]]:
            # logscale for frequencies, 0 start as 10**0 = 1
            frequencies = np.logspace(0.0,math.log((sampling_rate/2))/math.log(10), num = bands, dtype = np.float32)
            enc_d = []
            for k in range(bands):
                enc_d.append(math.sin(frequencies[k]*math.pi*xd))
                enc_d.append(math.cos(frequencies[k]*math.pi*xd))
            enc_d.append(xd)
            encoding.extend(enc_d)
        encodings.append(encoding)
    return encodings