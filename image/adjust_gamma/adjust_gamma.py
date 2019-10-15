import tensorflow as tf

def adjust_gamma(image, gamma=1, gain=1):
  """
  Performs Gamma Correction on the input image.
  This function transforms the input image pixelwise according to the
  equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.
  """
  return image