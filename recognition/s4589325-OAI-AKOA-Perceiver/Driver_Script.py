
# Import Perceiver file
import OAI_AKOA_Perceiver
import Parameters

# Libraries needed for data importing
import os
from PIL import Image
import math
import numpy as np
import random
import PIL
import itertools

# ##### Import Data #####

# Parameters
SAVE_DATA			= False
TEST_TRAINING_SPLIT	= 0.7
dataDirectory		= '../../../AKOA_Analysis/'

def save_data(dir):
	print("Saving Data...")

	print("Data Directory:", dir)

	# Need to sort data by patient so that we aren't leaking data between training and validation sets
	allPics = [dataDirectory + f for f in os.listdir(dataDirectory)]

	patients = [[0] * 2 for _ in range(len(allPics))]
	print("Number of Patients:", len(patients))

	i = 0
	for pic in allPics:
		# Get unique id for each patient
		pic = pic.split('OAI')[1]
		baseline_num_str = pic.split('de3d1')[0].split('BaseLine')[1]
		baseline_num = int(''.join(c for c in baseline_num_str if c.isdigit()))
		pic_num = pic.split('.nii.gz_')[1].split('.')[0]
		initial_id = pic.split('_')[0] + '_BaseLine_' + str(baseline_num) + '.' + pic_num
		patients[i][0] = initial_id
		i = i + 1

	# Assign each patient left or right status (slow code)
	ii = 0
	for i in patients:
		if ('right' in allPics[ii].lower()) or ('R_I_G_H_T' in allPics[ii]):
			patients[ii][1] = 1
		else:
			patients[ii][1] = 0
		ii += 1

	print('Right Knees:', sum([i[1] for i in patients]))
	print('Left Knees:', len(patients) - sum([i[1] for i in patients]))

	# Sort by substring
	patients = [list(i) for j, i in itertools.groupby(sorted(patients))]

	# Reduce number of images used
	patients = patients[0:math.floor(len(patients) * 0.25)]

	# Split data
	print("Splitting data into training and testing sets")
	patients_train	= patients[0:math.floor(len(patients) * TEST_TRAINING_SPLIT)]
	patents_test	= patients[math.floor(len(patients) * TEST_TRAINING_SPLIT):-1]

	# Remove extra axis added by the above sorting line
	patients_train	= [item for sublist in patients_train for item in sublist]
	patients_test	= [item for sublist in patents_test for item in sublist]

	# Verify no leakage
	patients_train_array = [i[0] for i in patients_train]
	patients_train_array = np.array(patients_train_array)
	patients_test_array = [i[0] for i in patients_test]
	patients_test_array  = np.array(patients_test_array)
	print("Intersection: ", np.intersect1d(patients_train_array, patients_test_array))

	# Import/Load images
	print("Importing training images")
	xtrain = []
	ytrain = []
	for i in patients_train:
		for j in allPics:
			if i[0].split('.')[0] in j and i[0].split('.')[1] in j.split('de3')[1]:
				xtrain.append(np.asarray(PIL.Image.open(j).convert("L").resize(Parameters.INPUT_SHAPE)))
				ytrain.append(i[1])
				break
	print("Importing testing images")
	xtest = []
	ytest = []
	for i in patients_test:
		for j in allPics:
			if i[0].split('.')[0] in j and i[0].split('.')[1] in j.split('de3')[1]:
				xtest.append(np.asarray(PIL.Image.open(j).convert("L").resize(Parameters.INPUT_SHAPE)))
				ytest.append(i[1])
				break

	# Normalize the data to [0,1]
	print("Normalizing data")
	xtrain = np.array(xtrain, dtype=float)
	xtest  = np.array(xtest, dtype=float)
	xtrain[:] /= 255
	xtest[:] /= 255
	ytrain = np.array(ytrain)
	ytest = np.array(ytest)

	# Shuffle xtrain
	train_order = list(range(0, len(xtrain)))
	random.shuffle(train_order)
	xtrain = xtrain[train_order]
	ytrain = ytrain[train_order]
	xtrain = np.squeeze(xtrain)
	ytrain = np.squeeze(ytrain)

	# Shuffle xtest
	test_order = list(range(0, len(xtest)))
	random.shuffle(test_order)
	xtest = xtest[test_order]
	ytest = ytest[test_order]
	xtest = np.squeeze(xtest)
	ytest = np.squeeze(ytest)
	
	# Save the data to local drive
	print("Saving data to disk")
	np.save('../../../xtrain', xtrain)
	np.save('../../../ytrain', ytrain)
	np.save('../../../xtest', xtest)
	np.save('../../../ytest', ytest)

# Save the data
if SAVE_DATA:
	save_data(dataDirectory)

# Load Data
print("Loading Data")
xtrain = np.load('../../../xtrain.npy')
ytrain = np.load('../../../ytrain.npy')
xtest = np.load('../../../xtest.npy')
ytest = np.load('../../../ytest.npy')

# Print data shape
print("xtrain shape:", xtrain.shape)
print("ytrain shape:", ytrain.shape)
print("xtest shape:", xtest.shape)
print("ytest shape:", ytest.shape)

# ##### Train and Evaluate Perceiver #####

# Create perceiver
perceiver = OAI_AKOA_Perceiver.Perceiver()

# Train Perceiver
[model_history, accuracy] = perceiver.train_perceiver(xtrain, ytrain, xtest, ytest)

print("\nAccuracy:", accuracy, "\n")

# Show summary
perceiver.summary()

# Plot
import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(model_history.history['binary_accuracy'], label='accuracy')
plt.plot(model_history.history['val_binary_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower left')
plt.savefig('Accuracy_Plot')

plt.figure(1)
plt.ylim([0, 1])
plt.plot(model_history.history['loss'], label='loss')
plt.plot(model_history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.savefig('Loss_Plot')
