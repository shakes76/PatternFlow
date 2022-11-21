import os
import cv2 as cv
import SimpleITK as itk
from skimage import io


# Preprocesses the data and stores the preprocessed data into the directories.
# And returns train_x, train_y, test_x, and test_y respectively.
def data_preprocessing():
    print("Data preprocessing started")
    print("Data loading started")
    # Paths of the training, validation, and test images of ISIC dataset.

    training_images_path = 'E:/Uni/COMP3710/ISIC-2017_Training_Data/'
    training_images = os.listdir(training_images_path)

    training_ground_truth_images_path = 'E:/Uni/COMP3710/ISIC-2017_Training_Part1_GroundTruth/'
    training_ground_truth_images = os.listdir(training_ground_truth_images_path)

    validation_images_path = 'E:/Uni/COMP3710/ISIC-2017_Validation_Data/'
    validation_images = os.listdir(validation_images_path)

    validation_ground_truth_images_path = 'E:/Uni/COMP3710/ISIC-2017_Validation_Part1_GroundTruth/'
    validation_ground_truth_images = os.listdir(validation_ground_truth_images_path)

    test_images_path = 'E:/Uni/COMP3710/ISIC-2017_Test_v2_Data/'
    test_images = os.listdir(test_images_path)

    test_ground_truth_images_path = 'E:/Uni/COMP3710/ISIC-2017_Test_v2_Part1_GroundTruth/'
    test_ground_truth_images = os.listdir(test_ground_truth_images_path)
    print("Data loading ended")
    print("Data saving started")
    # Path for saving the data. Change the paths to where you are going to save the data.
    save_train_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/train_data/'
    save_training_ground_truth_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/train_ground_truth_data/'
    save_validation_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/validation_data/'
    save_validation_ground_truth_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/validation_ground_truth_data/'
    save_test_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/test_data/'
    save_test_ground_truth_data = 'E:/Uni/COMP3710/Assignment/PatternFlow/recognition/MinsooHan_42570893/test_ground_truth_data/'
    print("Data saving ended")
    # Create directories for saving.

    if not os.path.exists(save_train_data):
        os.mkdir(save_train_data)
    if not os.path.exists(save_training_ground_truth_data):
        os.mkdir(save_training_ground_truth_data)
    if not os.path.exists(save_validation_data):
        os.mkdir(save_validation_data)
    if not os.path.exists(save_validation_ground_truth_data):
        os.mkdir(save_validation_ground_truth_data)
    if not os.path.exists(save_test_data):
        os.mkdir(save_test_data)
    if not os.path.exists(save_test_ground_truth_data):
        os.mkdir(save_test_ground_truth_data)

    image_size = 256

    # Define functions for creating training, validation, and test images.

    def create_training_images():
        data = []
        for image in training_images:
            read_image = itk.ReadImage(os.path.join(training_images_path, image))
            image_array = itk.GetArrayFromImage(read_image)
            resized_array = cv.resize(image_array, (image_size, image_size))
            data.append([resized_array])
            io.imsave(save_train_data + image[:-4] + '.png', resized_array)
        return data

    def create_training_ground_truth_images():
        data = []
        for image in training_ground_truth_images:
            read_image = itk.ReadImage(os.path.join(training_ground_truth_images_path, image))
            image_array = itk.GetArrayFromImage(read_image)
            resized_array = cv.resize(image_array, (image_size, image_size))
            data.append([resized_array])
            io.imsave(save_training_ground_truth_data + image[:-4] + '.png', resized_array)
        return data

    def create_validation_images():
        data = []
        for image in validation_images:
            read_image = itk.ReadImage(os.path.join(validation_images_path, image))
            image_array = itk.GetArrayFromImage(read_image)
            resized_array = cv.resize(image_array, (image_size, image_size))
            data.append([resized_array])
            io.imsave(save_validation_data + image[:-4] + '.png', resized_array)
        return data

    def create_validation_ground_truth_images():
        data = []
        for image in validation_ground_truth_images:
            read_image = itk.ReadImage(os.path.join(validation_ground_truth_images_path, image))
            image_array = itk.GetArrayFromImage(read_image)
            resized_array = cv.resize(image_array, (image_size, image_size))
            data.append([resized_array])
            io.imsave(save_validation_ground_truth_data + image[:-4] + '.png', resized_array)
        return data

    def create_test_images():
        data = []
        for image in test_images:
            read_image = itk.ReadImage(os.path.join(test_images_path, image))
            image_array = itk.GetArrayFromImage(read_image)
            resized_array = cv.resize(image_array, (image_size, image_size))
            data.append([resized_array])
            io.imsave(save_test_data + image[:-4] + '.png', resized_array)
        return data

    def create_test_ground_truth_images():
        data = []
        for image in test_ground_truth_images:
            read_image = itk.ReadImage(os.path.join(test_ground_truth_images_path, image))
            image_array = itk.GetArrayFromImage(read_image)
            resized_array = cv.resize(image_array, (image_size, image_size))
            data.append([resized_array])
            io.imsave(save_test_ground_truth_data + image[:-4] + '.png', resized_array)
        return data

    print("Creating images started")
    # Create the images

    create_training_images()
    create_training_ground_truth_images()
    create_validation_images()
    create_validation_ground_truth_images()
    create_test_images()
    create_test_ground_truth_images()
    print("Creating images ended")
