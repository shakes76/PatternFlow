import tensorflow as tf

# data structures for dtype_limits function
_integer_types = (tf.int8, tf.uint8,
                tf.int16, tf.uint16,
                tf.int32, tf.uint32,
                tf.int64, tf.uint64)
_integer_ranges = {t: (t.min, t.max)
                for t in _integer_types}
dtype_range = {tf.bool: (False, True),
            tf.float16: (-1.0, 1.0),
            tf.float32: (-1.0, 1.0),
            tf.float64: (-1.0, 1.0)}
dtype_range.update(_integer_ranges)

def dtype_limits(image, clip_negative=False):
  "Return intensity limits, i.e. (min, max) tuple, of the image's dtype."
  imin, imax = dtype_range[image.dtype]
  if clip_negative:
      imin = 0
  return imin, imax

def adjust_gamma(image, gamma=1, gain=1):
  """
  Performs Gamma Correction on the input image.
  This function transforms the input image pixelwise according to the
  equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.
  """
  
  if gamma < 0:
    raise ValueError("Gamma should be a non-negative real number.")
  
  dtype = image.dtype

  scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

  out = ((image / scale) ** gamma) * scale * gain
  return tf.cast(out, dtype)