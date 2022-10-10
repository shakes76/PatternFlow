
import os 
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt

def image_show(file_path, images_paths, image_num):
    ''' Show the image_num'th image from file_path with printing out its shape'''
    image_array=sitk.ReadImage(os.path.join(file_path,images_paths[image_num]))
    print(sitk.GetArrayFromImage(image_array).shape)
    plt.imshow(sitk.GetArrayFromImage(image_array))
    plt.show() 

