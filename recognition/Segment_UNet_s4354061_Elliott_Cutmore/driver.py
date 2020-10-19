
""" Uncomment if segment.py not module not found: """
import sys
import os
# sys.path.append("H:\\COMP3710\\PatternFlow\\recognition\\Segment_UNet_s4354061_Elliott_Cutmore")
# sys.path.append("C:\\Windows\\System32")
from segment import *

if __name__ == "__main__":

    option = 2
    inspect_image_sizes = False
    subset=False


    # Black and white Binary segmentation
    num_classes = 2
    # how many images to run through net:
    batch_size = 4
    # How many cycles to train the net for:
    epochs = 100
    # where to load images from (input and targets)
    # data_dir = "H:\\COMP3710\\PatternFlow\\recognition/Segment_UNet_s4354061_Elliott_Cutmore"
    img_dir = "H:\\COMP3710\\ISIC2018_Task1-2_Training_Input_x2"
    seg_dir = "H:\\COMP3710\\ISIC2018_Task1_Training_GroundTruth_x2"
    # Where to save trained model and checkpoints to from training
    save_model_path = ".\\model_1"
    save_checkpoint_path = ".\\training_1\\cp.ckpt"
    save_history_path = ".\\history_1_pickle"
    # Get all filename's from the paths of inputs and targets specified:
    input_img_paths, target_img_paths = \
        get_img_target_paths(img_dir, seg_dir)

    # This outputs histograms ao the image sizes in the paths given:
    # This is useful when determining what image size should be used
    # when training the UNet later.
    if inspect_image_sizes:
        print("Getting images sizes...")
        sizes = get_img_sizes(input_img_paths)
        max_w = max(sizes, key=lambda item: item[0])
        max_h = max(sizes, key=lambda item: item[1])
        min_w = min(sizes, key=lambda item: item[0])
        min_h = min(sizes, key=lambda item: item[1])
        print("max_h: ", str(max_h), "\nmax_w: ",
              str(max_w), "\nmin_h: ", str(min_h),
              "\nmin_w: ", str(min_w))

        print("Inspecting images and creating histograms...")
        inspect_images(sizes)

    # Use input images of (width, height) = (256, 256)
    # and 3 colour channels (RGB)
    img_dims = (256, 256, 3)

    # ## CHANGE THIS LATER!!!!!
    if subset:
        input_img_paths = input_img_paths[0:100]
        target_img_paths = target_img_paths[0:100]

    print("Getting a train, validation and testing split from data...")
    # Get a training, validation and test split of the data:
    train_input, train_target, val_input, val_target, test_input, test_target = \
        train_val_test_split(0.2, input_img_paths, target_img_paths)
    # print the size of each set:
    print("Array lengths:\nTrain: ", str(len(train_input)),
          "\nTest: ", str(len(test_input)), "\nVal: ", str(len(val_input)))

    # Since the dataset is so large (3GB) it will not all fit into RAM.
    # Therefore, a data generator must be used to train this model:
    # They load a small batch (batch_size) of images from storage to give
    # to the model for each training iteration within a training epoch
    print("Creating generators for model...")
    train_gen = create_generator(train_input, train_target, img_dims, batch_size, num_classes)
    val_gen = create_generator(val_input, val_target, img_dims, batch_size, num_classes)
    test_gen = create_generator(test_input, test_target, img_dims, batch_size, num_classes)

    # Check that generators work loading in images and visualise some:
    print("Checking generators...")
    check_generator(train_gen, img_dims, batch_size, num_classes, visualise=False)
    check_generator(val_gen, img_dims, batch_size, num_classes, visualise=False)
    check_generator(test_gen, img_dims, batch_size, num_classes, visualise=False)

    """ Option 1 - train a new net: """
    if option == 1:
        # Create a UNet model instance:
        print("Creating a model...")
        model = create_model(img_dims, num_classes)
        model.summary()

        # Train the new UNet model and check where its going to be saved:
        print("save_path: %s" % save_checkpoint_path)
        print("save_model_filename: %s" % save_model_path)
        history = train_model(train_gen, val_gen, model, epochs=epochs,
                              save_model_path=save_model_path,
                              save_checkpoint_path=save_checkpoint_path,
                              save_history_path=save_history_path)

        print("Plotting training history...")
        training_plot(history)
        print("Evaluating a test set/generator")
        test_preds, test_loss, test_acc = evaluate(test_gen, model)
        print("Test set size: ", len(test_input))
        print("Test loss: ", test_loss)
        print("Test accuracy: ", test_acc, '\n')

        print("Collating results...")
        results(test_input, test_target, test_preds, 5, img_dims, num_classes, visualise=True)

    """ Option 2 - load pre-trained net: """
    if option == 2:

        load_path = ".\\model_2"
        history_load_path = ".\\history_2_pickle"
        model = load_model(load_path)
        history = load_history(history_load_path)

        print("Plotting training history...")
        training_plot(history)
        print("Evaluating a test set/generator")
        test_preds, test_loss, test_acc = evaluate(test_gen, model)
        print("Test set size: ", len(test_preds))
        print("Test loss: ", test_loss)
        print("Test accuracy: ", test_acc, '\n')

        print("Collating results...")
        results(test_input, test_target, test_preds, 5, img_dims, num_classes, visualise=True)

