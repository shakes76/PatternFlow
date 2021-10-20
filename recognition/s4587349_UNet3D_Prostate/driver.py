

"""
Patients had from 1 to 8 MRI scans, a week apart. As scans for a given patient are expected
 to be similar each patients scans have been considered as one sample.
 All up there are 38 patients, and these have been distributed between training, validation and
 testing at 27:7:4 with the number of images at 158:35:18.
"""



# Data sources
X_train_dir = 'D:\prostate\mr_train'
X_validate_dir = 'D:\prostate\mr_validate'
X_test_dir = 'D:\prostate\mr_test'
# Label sources
Y_train_dir = 'D:\prostate\label_train'
Y_validate_dir = 'D:\prostate\label_validate'
Y_test_dir = 'D:\prostate\label_test'













def main():
    """ """
    pass


if __name__ == '__main__':
    main()