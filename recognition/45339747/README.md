# README (Jonathan Godbold, 45339747)
Algorithm implemented for Task 2: Classify laterality (left or right sided knee) of the OAI AKOA knee data set having a minimum accuracy of 0.9 on the test set. [Easy Difficulty]

Main executable can be run from execute_script_45339747.py which calls:
* generator_script_45339747.py
* layers_script_45339747.py
* results_script_45339747.py

The script generator_script_45339747.py imports the AKOA OASIS dataset.
* Function generate_paths() parses the dataset and adds every unique patient ID to a list. Split the data into three sets: training set, validation set and testing set. Since there are 101 total patients, the data was split into sets of 70, 20 and 21, where there were 70 patients for testing, 20 patients for validation and 21 patients for testing. This was required to prevent data leakage when building the model. 
* Function generate_sets() parses the dataset and uses the list of unique IDs to get which images belong to which patient and add them to their respective set depending on their position in the unique list of IDs (train, validate, test).
