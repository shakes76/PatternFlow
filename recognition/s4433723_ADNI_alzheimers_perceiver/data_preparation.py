import os
import sys

IMG_DIR = "AKOA_Analysis/"
CLASSES = ("Left", "Right")

def underscore_modify(string):

    modified_string = ""
    for char in string:
        modified_string += f"_{char}"
    return modified_string

def create_sorted_data_directory(img_dir, classes):

    # create subdirectories in dataset folder
    for class_name in classes:
        try:
            os.mkdir(f"datasets/{class_name}")
        except OSError as e:
            print(e)

    # move files from input dataset into datasets folder, in class sub-dirs
    for file_name in os.listdir(IMG_DIR):
        print(file_name)
        file_path = os.path.join(IMG_DIR, file_name)
        print(file_path)
        for class_name in classes:
            if class_name in file_name \
                    or class_name.upper() in file_name \
                    or underscore_modify(class_name.upper()) in file_name\
                    or class_name.lower() in file_name:
                os.rename(file_path, f"datasets/{class_name}/{file_name}")



if __name__ == "__main__":
    # handle input args to load into dataset directory etc.
    input_args = sys.argv

    create_sorted_data_directory(IMG_DIR, CLASSES)
    pass