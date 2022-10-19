import os
from PIL import Image

# Constants
IMAGE_DIM = 640

def resize_images(folder_path, dim, verbose = False):
    # Loop through the designated folder and look for images to resize
    for dirname, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            if '.jpg' in file_path or 'segmentation.png' in file_path:
                # Resize and save images
                resized_img = (Image.open(file_path)).resize((dim, dim))
                if verbose:
                    print(file_path[:-4] + f'_resized_{dim}' + file_path[-4:])
                resized_img.save(file_path[:-4] + f'_resized_{dim}' + file_path[-4:])

def main():
    resize_images("D:\GitHub\COMP3710 DEMO 3\ISIC", IMAGE_DIM, verbose=True)

if __name__ == "__main__":
    main()