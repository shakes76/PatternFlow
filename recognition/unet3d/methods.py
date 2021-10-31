import os
import re
import nibabel as nib
from scipy import ndimage
import tensorflow as tf



def read_filename_from(directory):
    """"
    Read image names from target directory and sort by case no. then week no.
    """
    files = os.listdir(directory)
    files = [fname[:-12] for fname in files if fname.endswith('nii.gz')]
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    return files



def train_val_test_split(directory):
    """
    Split images into train/validatiion/test sets on paticient basis
    """
    files = read_filename_from(directory)
    case_dict = {}
    # classify images by paticient into dict
    for f in files:
        if f[:8] not in case_dict:
            case_dict[f[:8]] = [f]
        else:
            case_dict[f[:8]].append(f)
    train = []
    val = []
    test = []
    for key, cases in case_dict.items():
        num = len(cases)
        # if one patient has only one image then allocate to training set
        if num == 1:
            train.append(cases[0])
        # if one patient has two images then allocate to training set and val set
        if num == 2:
            train.append(cases[0])
            val.append(cases[1])
        # if one patient has more two  images then allocate one to val set and one to test set, and the rest allocated to training set.
        if num >=3:
            train += cases[:-2]
            val.append(cases[-2])
            test.append(cases[-1])
    print("Numbers of cases for train, validation and test are: ", len(train), len(val), len(test), ", respectively")
    return train, val, test

def get_filepath_from(directory):
    """
    Get complete file paths for each set
    """

    train, val, test = train_val_test_split(directory)

    return [os.path.join(directory, fname) for fname in train], [os.path.join(directory, fname) for fname in val], [os.path.join(directory, fname) for fname in test]
            


def read_nii_file(path):
    """
    read nii image as numpy data
    """
    return nib.load(path).get_fdata()

    
def normalize(volume):
    """
    Scale and normalize img data
    """
    min, max = 0, 500
    # min-max scalling
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume-min)/(max-min)
    return volume

def resize_volume(img):
    """
    Resize img to desired shape
    """
    desired_shape = (256, 256, 128)
    
    if img.shape == desired_shape:
        pass
    else: 
        # Get current depth
        current_width, current_height, current_depth = img.shape
        # Set the desired depth
        desired_width, desired_height, desired_depth = desired_shape
        # Compute depth factor
        depth_factor = desired_depth/current_depth
        width_factor = desired_width / current_width
        height_factor = desired_height / current_height
        # Resize across z-axis with bilinear interpolation
        img = ndimage.zoom(img.astype(int), (width_factor, height_factor, depth_factor), order=1)
    return img

def resize_and_replace_volumes(imgs=[ "datasets/semantic_labels_anon/Case_019_Week1_SEMANTIC_LFOV.nii.gz", "datasets/HipMRI_study_complete_release_v1/semantic_MRs_anon/Case_019_Week1_LFOV.nii.gz"]):
     """
     Resize irregular img and replace the origin
     """
     for img_path in imgs:
        img = nib.load(img_path)
        data = img.get_fdata()
        if data.shape != (256, 256, 128):
            # resize volume to desired shape
            resized_data = resize_volume(data)
            # convert image to NIfTI1 format 
            resized_img = nib.Nifti1Image(resized_data.astype(int), img.affine, img.header)
            resized_img.header.get_data_shape()
            # save and replace the orign
            nib.save(resized_img, img_path)

        
def process_nii(path, norma = False):
    """
    Read nii.gz file; normalize data; apply agumentation; add channel with size 1;
    """
    volume = read_nii_file(path)
    if norma:
        volume = normalize(volume)
    # volume = resize_volume(volume)
    volume = tf.image.random_flip_up_down(volume, 3)
    volume = tf.expand_dims(volume, axis=3)
    return volume

# Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(tf.keras.utils.Sequence):
    """
    3D DataGenerator inherit from keras
    """
    
    def __init__(self, list_IDs, labels=None, batch_size=1, dim=(256,256,128), n_channels=1,
                 n_classes=None, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(tf.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # List of IDs (name of imgs)
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        seed = 42
        tf.random.set_seed(seed)
        'Updates indexes after each epoch'
        self.indexes = tf.range(len(self.list_IDs))
        if self.shuffle == True:
            self.indexes = tf.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = []
        Y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store mri case
            x = process_nii('./unet3d/dataset/semantic_MRs_anon/' + ID + '_LFOV.nii.gz', norma=True)
            X.append(x)

            # Store mri label
            y = process_nii('./unet3d/dataset/semantic_labels_anon/' + ID + '_SEMANTIC_LFOV.nii.gz')
            Y.append(y)

        X = tf.stack(X)
        Y = tf.stack(Y)
        return X, Y




