from os import listdir, path
import matplotlib.image as mpimg

class DataLoader():
    def __init__(self):
        self.train_set = []
    
    def load(self, parser):
        which_file = parser.get_training_path()
        which_file_abs_path = path.abspath(which_file)
        # Load all the images here through listdir
        for i in listdir(which_file_abs_path):
            source = "{}/{}".format(which_file_abs_path, i)
            image_read = mpimg.imread(source)
            # Normalising value
            image_read = image_read * 2 - 1
            self.train_set.append(
                image_read
            )
    
    def get_dataset(self):
        return self.train_set