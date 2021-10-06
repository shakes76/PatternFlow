# ##### Setup #####

# Libraries needed for model
#import tensorflow as tf
#from tensorflow.keras import layers, models
import PIL

# Libraries needed for data importing
import os
import itertools
from PIL import Image

#print("Tensorflow Version:", tf.__version__)

# ##### Macros #####

BATCH_SIZE			= 32
TEST_TRAINING_SPLIT	= 0.8
IMG_WIDTH			= 260
IMG_HEIGHT			= 228
SEED				= 123
LATENT_SIZE			= 512 # Same as paper
LEARNING_RATE		= 0.001
WEIGHT_DECAY		= 0.0001
EPOCHS				= 50
DROPOUT_RATE		= 0.2
PATCH_DIM			= 2

# ##### Import Data #####

#dataDirectory = '../../../AKOA_Analysis'
dataDirectory = '../AKOA_Analysis/'

# Need to sort data by patient so that we aren't leaking data between training and validation sets
allPics = [dataDirectory + f for f in os.listdir(dataDirectory)]

patients = [[0] * 2 for _ in range(len(allPics))]

i = 0
for pic in allPics:
	# Get unique id for each patient
	pic = pic.split('OAI')[1]
	baseline_num_str = pic.split('de3d1')[0].split('BaseLine')[1]
	baseline_num = int(''.join(c for c in baseline_num_str if c.isdigit()))
	initial_id = pic.split('_')[0] + '_BaseLine_' + str(baseline_num)
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
print(ii)
#exit()

# Sort by substring
#patients = [list(i) for j, i in itertools.groupby(sorted(patients))]

# Split data

# Remove extra axis added by the above sorting line
#patients = [item for sublist in patients for item in sublist]

print(patients)
print(len(patients))

print(sum([i[1] for i in patients]))
sum = 0
for i in patients:
	sum = sum + i[1]
print("Sum:", sum)

exit()
# Import Images
xtrain = []
ytrain = []
ii = 0
for i in patients:
	for j in allPics:
		if i[0] in j:
			print(ii)
			xtrain.append(PIL.Image.open(j))
			ytrain = i[1]
			ii = ii + 1




# Import actual data now


# Split into training/test sets
#train_patients = 
#test_patients  = 

''' # Cannot use this code because it leaks data between training/test sets
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
	validation_split=0.2,
	subset="training",
	seed=SEED,
	label_mode=None,
	image_size=(IMG_WIDTH, IMG_HEIGHT),
	batch_size=BATCH_SIZE,
	color_mode='grayscale'
)
dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
	validation_split=0.2,
	subset="validation",
	seed=SEED,
	label_mode=None,
	image_size=(IMG_WIDTH, IMG_HEIGHT),
	batch_size=BATCH_SIZE,
	color_mode='grayscale'
)
'''

exit()

# Normalize the data to [0,1]
dataset_train = dataset_train.map(lambda a: a / 255.0)
dataset_validation = dataset_validation.map(lambda a: a / 255.0)

# Show some info on the dataset
print(type(dataset_train))
print(len(dataset_train))
print(type(dataset_validation))
print(len(dataset_validation))


