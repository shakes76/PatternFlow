"""REFERENCES:
   - (1) https://keras.io/examples/vision/super_resolution_sub_pixel/
"""
import matplotlib.pyplot as plt
from dataset import input_downsample, input_process
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import os
import tensorflow as tf
from constants import target_image_width, upsample_factor, test_path, checkpoint_loc
from modules import Model
from dataset import collect_test_images

comp_images_path = "comp_images"

def load_model():
  '''
  Load the trained model.
  '''
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
  #pad the image to square
  original_image = tf.image.pad_to_bounding_box(original_image, 0, 0, target_image_width, target_image_width)
  #expand to feed into input_process and input_downsample
  original_image = np.expand_dims(original_image, axis=0)

  #get downsampled image
  test_image = input_downsample(original_image, target_image_width, upsample_factor)
  
  #upsample test_image with bicubic method
  scaled = np.squeeze(test_image, axis=0)
  scaled = array_to_img(tf.image.resize(scaled, (target_image_width, target_image_width), method='bicubic'))
  
  #process to one colour channel
  original_image = input_process(original_image)
  original_image = np.squeeze(original_image, axis=0)

  #get prediction
  model_output = model.predict(test_image)

  #compute psnr of outputs
  model_psnr = tf.image.psnr(original_image, img_to_array(model_output), 255)
  scaled_psnr = tf.image.psnr(original_image, img_to_array(scaled), 255)

  #print or return information
  original_image = array_to_img(original_image)
  if (multi):
    return {"Original Image":original_image, "Bicubic Resizing":scaled, "Model Output":model_output}, (model_psnr, scaled_psnr)
  print(f"Model PSNR: {model_psnr}")
  print(f"Scaled PSNR: {scaled_psnr}")
  original_image.save(os.path_join(comp_images_path, 'original_image.png'))
  scaled.save(os.path_join(comp_images_path,'bicubic.png'))
  model_output.save(os.path_join(comp_images_path,'model_output.png'))
  
def produce_multiple_comparison_plot(model, test_path, num_comp=5):
  '''
  Creats a plot 3xnum_comp to compare output image, bicubic rezizing and model output
  :param model: the trained model
  :param test_path: the path to the test images
  :num_comp: the number of samples to compare
  '''
  cols, rows = num_comp, 3
  figure = plt.figure(figsize=(13, 22), dpi=80)
  for i in range(num_comp):
    images, (model_psnr, scaled_psnr) = produce_comparison_plot(model, test_path[i], True)
    
    for j, (title, img) in enumerate(images.items()):
      figure.add_subplot(cols, rows, 3 * i + j + 1)
      if (i == 0):
        plt.title(title)
      if ("Original" not in title):
        plt.xlabel(f"PSNR: {model_psnr if 'Model' in title else scaled_psnr}")
      plt.imshow(img, cmap='gray')
  figure.savefig(f"multi_comp_{num_comp}")        




model = load_model()
test = collect_test_images(test_path)

produce_multiple_comparison_plot(model, test)
