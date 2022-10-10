import os
import sys
import zipfile
import requests

dataset_location = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
dataset_directory = "dataset/"
dataset_zip_name = "keras_png_slices_data.zip"
dataset_folder_name = "keras_png_slices_data/"

# Download the dataset, if it hasn't already been downloaded
def download_dataset():
    # Create the dataset directory
    if not os.path.isdir(dataset_directory):
        os.mkdir(dataset_directory)

    # Download dataset zip
    print(f"Downloading dataset into ./{dataset_directory}{dataset_zip_name}")
    if not os.path.exists(dataset_directory + dataset_zip_name):
        response = requests.get(dataset_location, stream=True)
        total_length = response.headers.get("content-length")

        with open(dataset_directory + dataset_zip_name, "wb") as f:
            # Show download progress bar (doesn't work on non-Unix systems)
            # Adapted from https://stackoverflow.com/a/15645088
            if total_length is None or not os.name == "posix":
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)))
                    sys.stdout.write(" %d%%" % int(dl / total_length * 100))
                    sys.stdout.flush()
            print()

        print("Dataset downloaded.\n")
    else:
        print("Dataset already downloaded.\n")

# Unzip the dataset
def unzip_dataset():
    print(f"Extracting dataset into ./{dataset_directory}{dataset_folder_name}")
    if not os.path.isdir(dataset_directory + dataset_folder_name):
        with zipfile.ZipFile(dataset_directory + dataset_zip_name) as z:
            z.extractall(path=dataset_directory)
        print("Dataset extracted.\n")
    else:
        print("Dataset already extracted.\n")
