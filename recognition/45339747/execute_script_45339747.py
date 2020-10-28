print("Running...")
from generator_script_45339747 import *

# Generate the paths for each file in each section (train, validate, test).
subset_path_AKOA, train_list, validate_list, test_list = generate_paths()

# Split the dataset.
train_images_src, validate_images_src, test_images_src = generate_sets(train_list, validate_list, test_list, subset_path_AKOA)

# Load the data as tensors.
train_images, validate_images, test_images = loadData(train_images_src, validate_images_src, test_images_src)

# Retrieve the labels for the data.
train_images_y, validate_images_y, test_images_y = loadLabels(train_images_src, validate_images_src, test_images_src)

# Normalize the X-data and load the labels as tensors.
train_images, validate_images, test_images, train_images_y, validate_images_y, test_images_y = formatData(train_images, validate_images, test_images, train_images_y, validate_images_y, test_images_y)

print("All images loaded, building model...")

# Import Libraries.
import tensorflow as tf
from layers_script_45339747 import *

# Build the model.
model = buildNetwork(train_images[0].shape)

# Compile and run the model, print the final metric.
compile_and_run(model, 5, 20, train_images, train_images_y, validate_images, validate_images_y)

# End of script. Please see involved scripts for more information regarding the methods.