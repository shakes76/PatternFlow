# rank_order

This is a tensorflow implementation of skimage.filters.rank_order algorithm.<br>
The module computes an image of the same shape as an input image,
where each pixel is the index of the original pixel value.<br>
the indexes are in the ascending order i.e. from 0 to n-1, where n
is the number of unique values in original image.

## Usage:
`from rank_order import rank_order`<br>
`processed_file = rank_order(<path_to_jpeg_image>)`
returns a tuple, containing filtered image and the original image.

## Example
run the main.py script as following: `python main.py`<br>
when prompted - specify the location of
image file to convert (eg. "images/raw.jpg")<br>
the resulting image will be saved as "image/example_output.jpg" file.

## Example outputs
original image:<br> ![raw_image](images/raw.jpg)<br>
rank_order image:<br> ![processed_image](images/example_output.jpg)

*(Photo by Colin Horn on Unsplash)*

