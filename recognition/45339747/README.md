# README (Jonathan Godbold, 45339747)
Algorithm implemented for Task 2: Classify laterality (left or right sided knee) of the OAI AKOA knee data set having a minimum accuracy of 0.9 on the test set. [Easy Difficulty]

Main executable can be run from execute_script_45339747.py which calls:
* generator_script_45339747.py
* layers_script_45339747.py
* results_script_45339747.py

The script generator_script_45339747.py imports the AKOA OASIS dataset.
* Function generate_paths() parses the dataset and adds every unique patient ID to a list. Split the data into three sets: training set, validation set and testing set. Since there are 101 total patients, the data was split into sets of 70, 20 and 21, where there were 70 patients for testing, 20 patients for validation and 21 patients for testing. This was required to prevent data leakage when building the model. 
* Function generate_sets() parses the dataset and uses the list of unique IDs to get which images belong to which patient and add them to their respective set depending on their position in the unique list of IDs (train, validate, test).
* Function load_data() loads the respective images for the files in the lists above using NumPy. Note this function loads 3 of the same images due to the image format of the MRI scans supplied (this issue is resolved in the following function).
* Function loadLables() creates a set of labels for each training, validation and test sets. It does this by reading through the list of files and if the file name contains RIGHT, a 1 is allocated or if the file name contains LEFT, a 0 is allocated.
* Function formatData() normalizes and formats the data. Begin by dividing each of the training, validation and testing sets by 255, so that the sigmoid activation function in the model can interpret these values. Convert each of the Y labels in NumPy arrays as they are currently just default lists. Each of the X and Y sets were then converted to tensors for training, validation and testing.
