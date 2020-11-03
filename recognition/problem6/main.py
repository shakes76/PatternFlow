import sys

from load_data import *

if __name__ == "__main__":
    path = sys.argv[1]
    
    path = os.path.join(path, 'keras_png_slices_data')
    path = os.path.join(path, 'keras_png_slices_seg_train')
    
    image_paths = load_paths(path)

    image_data = tf.data.Dataset.from_tensor_slices(image_paths)

    image_data = image_data.map(load_images)
    
