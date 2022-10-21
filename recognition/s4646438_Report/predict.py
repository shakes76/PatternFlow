"""REFERENCES:
   - (1) https://keras.io/examples/vision/super_resolution_sub_pixel/
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from dataset import input_downsample
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from IPython.display import display
import numpy as np
import os
import tensorflow as tf
from constants import target_image_width, upsample_factor, test_path, checkpoint_loc
from modules import Model
from dataset import collect_test_images

comp_images_path = "comp_images"

def load_model():
  model = Model(upsample_factor)
  model.load_weights(checkpoint_loc)
  return model

def produce_comparison_plot(model, input_path, multi:bool):
  '''
  Produces a set of images. Original image, a scaled up version using bicubic resizing and the output from my model
  If multi is true, the images will be returned. If not they will be saved to files.
  '''
  #load in the test image, pad and resize it to pass into the model
  original_image = img_to_array(load_img(input_path))
  original_image = tf.image.pad_to_bounding_box(original_image, 0, 0, target_image_width, target_image_width)
  test_image = np.expand_dims(original_image, axis=0)
  original_image = array_to_img(original_image)
  test_image = input_downsample(test_image, target_image_width, upsample_factor)
  scaled = np.squeeze(test_image, axis=0)
  scaled = array_to_img(tf.image.resize(scaled, (target_image_width, target_image_width), method='bicubic'))
  model_output = model.predict(test_image)
  if (multi):
    return {"Original Image":original_image, "Bicubic Resizing":scaled, "Model Output":model_output}

  original_image.save(os.path_join(comp_images_path, 'original_image.png'))
  scaled.save(os.path_join(comp_images_path,'comp_images/bicubic.png'))
  model_output.save(os.path_join(comp_images_path,'comp_images/model_output.png'))
  
def produce_multiple_comparison_plot(model, test_path, num_comp=5):
  cols, rows = num_comp, 3
  figure = plt.figure(figsize=(20, 30), dpi=80)
  for i in range(num_comp):
    images = produce_comparison_plot(model, test_path[i], True)
    for j, (title, img) in enumerate(images.items()):
      figure.add_subplot(cols, rows, 3 * i + j + 1)
      if (i == 0):
        plt.title(title)
      plt.axis('off')
      plt.imshow(img, cmap='gray')
  figure.savefig(f"multi_comp_{num_comp}")        




model = load_model()
test = collect_test_images(test_path)

produce_multiple_comparison_plot(model, test)
