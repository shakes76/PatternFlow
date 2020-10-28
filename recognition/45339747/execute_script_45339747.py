print("Running...")
from generator_script_45339747 import *

#subset_path_AKOA, train_list, validate_list, test_list = generate_paths()

#train_images_src, validate_images_src, test_images_src = generate_sets(train_list, validate_list, test_list, subset_path_AKOA)

#train_images, validate_images, test_images = loadData(train_images_src, validate_images_src, test_images_src)

#train_images_y, validate_images_y, test_images_y = loadLabels(train_images_src, validate_images_src, test_images_src)

#train_images, validate_images, test_images, train_images_y, validate_images_y, test_images_y = formatData(train_images, validate_images, test_images, train_images_y, validate_images_y, test_images_y)

# Import Libraries.
import tensorflow as tf
from layers_script_45339747 import *

model = buildNetwork()