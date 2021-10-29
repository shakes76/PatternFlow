# Rescale Intensity

## Description:
The function takes an array as input and rescales the values in it based on the parameters passed to it. The options available for the input range and out range are the same. It clips the input array based on the chosen parameters and then rescales the array based on the chosen out_range. The output min and max The output is a numpy array with the shape and data type but scaled intensities. 


## Supported rescale types:

1. Tuple with min and max: It rescales the values in the array using that range
2. "image": Uses the input's max and min to rescale the values in the array
3. "dtype": Uses the input's dtype to fix max and min values which is then used to rescale the values in the array
4. dtype-name: Uses the min and max of the dtype specified. 


* numpy.bool_: (False, True),
* numpy.float16: (-1, 1),
* numpy.float32: (-1, 1),
* numpy.float64: (-1, 1),
* numpy.int16: (-32768, 32767),
* numpy.int32: (-2147483648, 2147483647),
* numpy.int64: (-9223372036854775808, 9223372036854775807),
* numpy.int64: (-9223372036854775808, 9223372036854775807),
* numpy.int8: (-128, 127),
* numpy.uint16: (0, 65535),
* numpy.uint32: (0, 4294967295),
* numpy.uint64: (0, 18446744073709551615),
* numpy.uint64: (0, 18446744073709551615),
* numpy.uint8: (0, 255),
* 'bool': (False, True),
* 'bool_': (False, True),
* 'float': (-1, 1),
* 'float16': (-1, 1),
* 'float32': (-1, 1),
* 'float64': (-1, 1),
* 'int16': (-32768, 32767),
* 'int32': (-2147483648, 2147483647),
* 'int64': (-9223372036854775808, 9223372036854775807),
* 'int8': (-128, 127),
* 'uint10': (0, 1023),
* 'uint12': (0, 4095),
* 'uint14': (0, 16383),
* 'uint16': (0, 65535),
* 'uint32': (0, 4294967295),
* 'uint64': (0, 18446744073709551615),
* 'uint8': (0, 255)


## Working:
As soon as the module is called, it builds the dictionary with supported data types and the range of values they support. The algorithm starts with setting an imin and imax value based on the parameters passed. If there is no range value passed, the input range is chosen as the image range and output range is chosen as the data type range. intensity_range returns the min and max values based on the parameters passed. These values are used to calculate the min and max for input and output. Once calculated, the array is clipped based on the input min and max values. The output is calculated by scaling based on the out min and max values. 

## Examples
1. rescale_intensity([1,2,3,4],out_range=(0,1)) -->  [0., 0.33333334, 0.66666669, 1.]
2. rescale_intensity(np.array([51, 102, 153], dtype=np.uint8)) -->  [0, 127, 255]
3. rescale_intensity(np.array([51, 102, 153], dtype=np.float16)) -->  [0, 0.5, 1]
4. rescale_intensity(np.array([51, 102, 153], dtype=np.float16),in_range=(0,102)) --> [0.5, 1.,  1.]
5. rescale_intensity(np.array([51, 102, 153], dtype=np.unit8),in_range=(0,102)) --> [127, 255, 255]
6. rescale_intensity(np.array([51, 102, 153], dtype=np.unit8),out_range=(0,102)) --> [0, 51, 102]


## Visualisations 

camera = data.camera()

camera=np.array(camera)

plt.imshow(camera)

![Before scaled](https://i.ibb.co/Zc6xvsh/download.png)

plt.imshow(rescale_intensity(camera,out_range=(0,5)))

![After being scaled](https://i.ibb.co/BVnVJMx/download-1.png)
