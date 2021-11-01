import os

# Silence extremely verbose tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import argparse
from scipy.ndimage import zoom
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from model import unet3d

# Formulate train, val, test file paths for "scans" and "labels"
# Stored as filepaths, as the generator will do the file reading
def get_nifti_files_in(directory):
	paths = []
	for file in os.listdir(directory):
		if file.endswith(".nii.gz"):
			paths.append(os.path.join(directory, file))
	
	# Sort so ordering is not file system order dependent
	return sorted(paths)

# Categorise input data into a form which represents the patient heirarchy, to not train and test on the same patient!
def generate_patient_data_heirarchy(data_paths):
	# Just incase patient IDs aren't continuous, i will use a dictionary mapping from id -> [[scan_path, label_path]]
	patient_data = {}
	
	for scan_path, label_path in data_paths:
		# Use scan filename to parse out patient ID
		scan_file = os.path.basename(scan_path)
		# Using scan formatting of "Case_[patient_id]_..."
		patient_id = int(scan_file.split("_")[1])
		
		# Simple dict, array population with first time checking
		if patient_id in patient_data:
			patient_data[patient_id].append([scan_path, label_path])
		else:
			patient_data[patient_id] = [[scan_path, label_path]]
		
	return patient_data

# Create Train, Val, Test split (move into utility function)
def train_val_test_split(train_split, val_split, test_split, patient_data):
	print("Splitting dataset using Train: " + str(train_split) + 
		  " Val: " + str(val_split) + " Test: " + str(test_split))

	assert train_split + val_split + test_split == 1.0

	# Data is not shuffled, so we have identical buckets each time. This means that across
	# Multiple runs of the driver.py, there is no possibility of cheating
	patient_ids = list(patient_data.keys())
	
	train_data = []
	val_data = []
	test_data = []
	
	# The method works by calculating the percent allocated after each allocation,
	# Then assigning more data to the bucket which is FURTHEST BELOW the desired
	# Allocation. This iterative method will give us a good allocation, which is close enough to the desired one.
	for patient_id in patient_ids:
		# Clamp to minimum value of 1 so we don't have to handle div by 0 on first iteration
		assigned_count = max(len(train_data) + len(val_data) + len(test_data), 1)
		
		# Calculated by (desired allocation percent - (no. allocated to that bucket / total allocated))
		train_displacement = train_split - (len(train_data) / assigned_count)
		val_displacement = val_split - (len(val_data) / assigned_count)
		test_displacement = test_split - (len(test_data) / assigned_count)
		
		# Allocate this patient to the bucket furthest from allocation (i.e. max displacement)
		if train_displacement > val_displacement and train_displacement > test_displacement:
			# Use extend so train_data is flat
			train_data.extend(patient_data[patient_id])
		elif val_displacement > test_displacement:
			val_data.extend(patient_data[patient_id])
		else:
			test_data.extend(patient_data[patient_id])
	
	return train_data, val_data, test_data

# Generator:
# Maximum scan voxel value in dataset was above 512 but below 1023 so i use closest power of two -> 1023, for normalisation
class Prostate3DGenerator(keras.utils.Sequence):
	def __init__(self, data_paths, batch_size, data_dimensions, class_count):
		self.data_paths = data_paths
		self.batch_size = batch_size
		self.data_dimensions = data_dimensions
		self.class_count = class_count
		
	def __len__(self):
		return int(np.floor(len(self.data_paths) / self.batch_size))
	
	# Used to infer data paths from generator object - used for test generator
	def get_data_paths(self):
		return self.data_paths

	def __getitem__(self, index):
		start_index = index * self.batch_size
		batch_data_paths = self.data_paths[start_index : start_index + self.batch_size]
		
		scans = np.empty((self.batch_size, *self.data_dimensions))
		labels = np.empty((self.batch_size, *self.data_dimensions, self.class_count), dtype=int)
		
		for dataIndex in range(len(batch_data_paths)):
			# Populate "scans"	
			scan_voxels = nib.load(batch_data_paths[dataIndex][0])
			# Downscale, so the shape is 2x smaller on each dimension
			np_scan_voxels = zoom(np.array(scan_voxels.dataobj), 0.5)
			scans[dataIndex] = tf.cast(np_scan_voxels / 1023.0, tf.float32)
			
			# Populate "labels"
			nibabel_voxels = nib.load(batch_data_paths[dataIndex][1])
			np_nibabel_voxels = np.array(nibabel_voxels.dataobj)
			# Since we don't want interpolated values for classes, just do a stride based downscale instead of spline interpolation
			downscaled_voxels = np_nibabel_voxels[0:256:2, 0:256:2, 0:128:2]
			labels[dataIndex] = keras.utils.to_categorical(downscaled_voxels, num_classes=self.class_count)
			
		return scans, labels

def strip_anomolous_data(data_paths, expected_scan_shape):
	# Scan for anomolous data, with unexpected dimensions (CSIRO dataset has one of these) 
	print("Processing Data for Anomolous Entries")
	anomaly_count = 0
	for idx in reversed(range(0, len(data_paths))):
		scan_path = data_paths[idx][0]
		scan_voxels = nib.load(scan_path)
		if (np.array(scan_voxels.dataobj).shape != expected_scan_shape):
			print("Anomaly Found: " + scan_path)
			anomaly_count += 1

			# Remove the anomaly from data_paths
			del data_paths[idx]

	print("Anomaly Scan Finished. " + str(anomaly_count) + " Anomalies Found And Removed.")

def build_generators(scans_directory, labels_directory, model_input_size, expected_scan_shape):
	scans = get_nifti_files_in(scans_directory)
	labels = get_nifti_files_in(labels_directory)

	# Make sure we have an equal number of scans and labels
	assert len(scans) == len(labels)

	# Zips the two "scans" and "labels" arrays together to produce [[scan_filename, label_filename], ...]
	data_paths = list(np.dstack((scans, labels))[0])

	print("Raw Labelled Scans: " + str(len(data_paths)))

	strip_anomolous_data(data_paths, expected_scan_shape)

	# Categorise data paths into patient based buckets
	patient_data = generate_patient_data_heirarchy(data_paths)

	print("Patient Count: " + str(len(patient_data.keys())))


	# Formulate training splits using patient categorised data
	train_paths, val_paths, test_paths = train_val_test_split(0.7, 0.15, 0.15, patient_data)

	# Sanity checks
	source_data_length = len(data_paths)
	assert len(train_paths) + len(val_paths) + len(test_paths) == source_data_length

	print("Train Count: " + str(len(train_paths)))
	print("Val Count: " + str(len(val_paths)))
	print("Test Count: " + str(len(test_paths)))


	# Initialize generators
	batch_size = 1

	train_generator = Prostate3DGenerator(train_paths, batch_size, model_input_size, class_count)
	val_generator = Prostate3DGenerator(val_paths, batch_size, model_input_size, class_count)
	test_generator = Prostate3DGenerator(test_paths, batch_size, model_input_size, class_count)
	
	return train_generator, val_generator, test_generator
	
def evaluate_model_with_upscale(model, test_scan_path, test_label_path, class_count):
	print(test_scan_path)
	print(test_label_path)

	# Load scan downscaled, to put into the model
	scan_voxels = nib.load(test_scan_path)
	# Downscale, so the shape is 2x smaller on each dimension
	np_scan_voxels = zoom(np.array(scan_voxels.dataobj), 0.5)
	scan_in = tf.cast(np_scan_voxels / 1023.0, tf.float32)

	# Load ground truth labels at full resolution to test against downscaled predictions
	nibabel_voxels = nib.load(test_label_path)
	np_nibabel_voxels = np.array(nibabel_voxels.dataobj)
	ground_truth_full = keras.utils.to_categorical(np_nibabel_voxels, num_classes=class_count)

	predicted_labels = model.predict(scan_in)
	print(predicted_labels.shape)

parser = argparse.ArgumentParser(description='A Simple UNet 3D Driver')
parser.add_argument("--mode", help="Mode the driver will run in.", default="train")
parser.add_argument("--epochs", help="The number of epochs to train if the driver is in train mode.", default="10")
args = parser.parse_args()

# Main
base_path = "data"
model_input_size = (128, 128, 64)
# Used to detect anomolous data
expected_scan_shape = (256, 256, 128)

# Background, Body, Bone, Bladder, Rectum, Prostate
class_count = 6

scans_directory = os.path.join(base_path, "semantic_MRs_anon")
labels_directory = os.path.join(base_path, "semantic_labels_anon")
train_generator, val_generator, test_generator = build_generators(scans_directory, labels_directory, model_input_size, expected_scan_shape)

if args.mode == "train":
	model = unet3d([32, 64, 128], model_input_size, class_count)
	#print(model.summary())

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

	callbacks = [
		keras.callbacks.ModelCheckpoint("prostate_model.h5", save_best_only=True)
	]

	# Train the model, doing validation at the end of each epoch.
	epochs = int(args.epochs)

	print("Starting Training!")

	train_data = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks)

	print("Training Complete!")

elif args.mode == "evaluate":
	model = unet3d([32, 64, 128], model_input_size, class_count)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
	model.load_weights("prostate_model.h5")

	test_scan_path, test_label_path = test_generator.get_data_paths()[0]
	evaluate_model_with_upscale(model, test_scan_path, test_label_path, class_count)