'''
Test script for running the segmentation module
'''

import sys
from ISIC_segment import train_model

# Constants for my machine
X_DATA_LOCATION = 'C:/Users/match/downloads/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'
Y_DATA_LOCATION = 'C:/Users/match/downloads/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'
EPOCHS = 6


def print_usage():
	'''
	Print a description of how to operate the script to the terminal
	'''
	print('USAGE: test_driver.py default')
	print('OR')
	print('USAGE: test_driver.py x_data_path y_data_path x_file_ext y_file_ext num_epochs batch_size')


def main():
	'''
	Main function for operating the script
	'''
	if len(sys.argv) == 2 and sys.argv[1] == 'default':
		train_model(X_DATA_LOCATION, Y_DATA_LOCATION, 'jpg', 'png', EPOCHS, 10)
	elif len(sys.argv) == 7:
		train_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
	else:
		print_usage()


if __name__ == '__main__':
	main()
