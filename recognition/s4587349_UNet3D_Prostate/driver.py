import os
import nibabel as nib
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

import unet_model as mdl
import support_methods as sm

"""
Sources
"""

"""
All images are 3D MRi's of shape (256, 256, 128) in nibabel format (*.nii.gz).
Data and labels are in numpy arrays, float64.
MRi voxel values vary from 0.0 upwards.
The labels have 6 classes, labelled from 0.0 to 5.0.
"""
dim = (256, 256, 128)
CLASSES = 6


def main():
    """ """
    """ Show reachable GPUs"""
    # print(tf.config.list_physical_devices(device_type='GPU'))    #todo remove

    """ 
    Patients had from 1 to 8 MRI scans, a week apart. As scans for a given
    patient are expected to be similar each patients scans have been considered as
    one sample. All up there are 38 patients, and these have been distributed
    between training, validation and testing at 27:7:4 with the number of images
    at 158:35:18.
    """


    """ DATA SOURCES"""
    # UNCOMMENT THE DIRECTORY STRUCTURE FOR THE SYSTEM YOU ARE WORKING ON.

    # """ Data Sources Windows D: """
    X_TRAIN_DIR = 'D:\\prostate\\mr_train'
    X_VALIDATE_DIR = 'D:\\prostate\\mr_validate'
    X_TEST_DIR = 'D:\\prostate\\mr_test'
    # Label sources
    Y_TRAIN_DIR = 'D:\\prostate\\label_train'
    Y_VALIDATE_DIR = 'D:\\prostate\\label_validate'
    Y_TEST_DIR = 'D:\\prostate\\label_test'

    """ Example data and label  """
    img_mr = (nib.load(X_TRAIN_DIR + '\\Case_004_Week0_LFOV.nii.gz')).get_fdata()
    img_label = (nib.load(Y_TRAIN_DIR + '\\Case_004_Week0_SEMANTIC_LFOV.nii.gz')).get_fdata()
    img_label2 = (nib.load(Y_TRAIN_DIR + '\\Case_011_Week7_SEMANTIC_LFOV.nii.gz')).get_fdata()

    """ Full data & label addresses in D: """
    image_train = sorted([os.path.join(os.getcwd(), 'D:\\prostate\\mr_train', x)
                          for x in os.listdir('D:\\prostate\\mr_train')])
    image_validate = sorted([os.path.join(os.getcwd(), 'D:\\prostate\\mr_validate', x)
                             for x in os.listdir('D:\\prostate\\mr_validate')])
    image_test = sorted([os.path.join(os.getcwd(), 'D:\\prostate\\mr_test', x)
                         for x in os.listdir('D:\\prostate\\mr_test')])
    label_train = sorted([os.path.join(os.getcwd(), 'D:\\prostate\\label_train', x)
                          for x in os.listdir('D:\\prostate\\label_train')])
    label_validate = sorted([os.path.join(os.getcwd(), 'D:\\prostate\\label_validate', x)
                             for x in os.listdir('D:\\prostate\\label_validate')])
    label_test = sorted([os.path.join(os.getcwd(), 'D:\\prostate\\label_test', x)
                         for x in os.listdir('D:\\prostate\\label_test')])

    """ Small test set D:"""
    data_small_train = sorted([os.path.join(os.getcwd(), 'D:\\p\\data', x)
                               for x in os.listdir('D:\\p\\data')])
    label_small_train = sorted([os.path.join(os.getcwd(), 'D:\\p\\label', x)
                                for x in os.listdir('D:\\p\\label')])
    data_small_validate = sorted([os.path.join(os.getcwd(), 'D:\\p\\data_validate', x)
                                  for x in os.listdir('D:\\p\\data_validate')])
    label_small_validate = sorted([os.path.join(os.getcwd(), 'D:\\p\\label_validate', x)
                                   for x in os.listdir('D:\\p\\label_validate')])
    data_small_test = sorted([os.path.join(os.getcwd(), 'D:\\p\\data_test', x)
                              for x in os.listdir('D:\\p\\data_test')])
    label_small_test = sorted([os.path.join(os.getcwd(), 'D:\\p\\label_test', x)
                               for x in os.listdir('D:\\p\\label_test')])

    """ Data Sources Windows C: """
    # # Data sources
    # X_TRAIN_DIR = 'C:\\prostate\\mr_train'
    # X_VALIDATE_DIR = 'C:\\prostate\\mr_validate'
    # X_TEST_DIR = 'C:\\prostate\\mr_test'
    # # Label sources
    # Y_TRAIN_DIR = 'C:\\prostate\\label_train'
    # Y_VALIDATE_DIR = 'C:\\prostate\\label_validate'
    # Y_TEST_DIR = 'C:\\prostate\\label_test'

    # """ Full data & label addresses in C: """
    # image_train = sorted([os.path.join(os.getcwd(), 'C:\\prostate\\mr_train', x)
    #                       for x in os.listdir('C:\\prostate\\mr_train')])
    # image_validate = sorted([os.path.join(os.getcwd(), 'C:\\prostate\\mr_validate', x)
    #                          for x in os.listdir('C:\\prostate\\mr_validate')])
    # image_test = sorted([os.path.join(os.getcwd(), 'C:\\prostate\\mr_test', x)
    #                      for x in os.listdir('C:\\prostate\\mr_test')])
    #
    #
    # label_train = sorted([os.path.join(os.getcwd(), 'C:\\prostate\\label_train', x)
    #                       for x in os.listdir('C:\\prostate\\label_train')])
    # label_validate = sorted([os.path.join(os.getcwd(), 'C:\\prostate\\label_validate', x)
    #                          for x in os.listdir('C:\\prostate\\label_validate')])
    # label_test = sorted([os.path.join(os.getcwd(), 'C:\\prostate\\label_test', x)
    #                      for x in os.listdir('C:\\prostate\\label_test')])

    # """ Small test set C:"""
    # data_small_train = sorted([os.path.join(os.getcwd(), 'C:\\p\\data', x)
    #                            for x in os.listdir('C:\\p\\data')])
    # label_small_train = sorted([os.path.join(os.getcwd(), 'C:\\p\\label', x)
    #                             for x in os.listdir('C:\\p\\label')])
    # data_small_validate = sorted([os.path.join(os.getcwd(), 'C:\\p\\data_validate', x)
    #                               for x in os.listdir('C:\\p\\data_validate')])
    # label_small_validate = sorted([os.path.join(os.getcwd(), 'C:\\p\\label_validate', x)
    #                                for x in os.listdir('C:\\p\\label_validate')])
    # data_small_test = sorted([os.path.join(os.getcwd(), 'C:\\p\\data_test', x)
    #                           for x in os.listdir('C:\\p\\data_test')])
    # label_small_test = sorted([os.path.join(os.getcwd(), 'C:\\p\\label_test', x)
    #                            for x in os.listdir('C:\\p\\label_test')])

    """ Data sources Cluster """
    # # Data sources
    # X_TRAIN_DIR = 'prostate/mr_train'
    # X_VALIDATE_DIR = 'prostate/mr_validate'
    # X_TEST_DIR = 'prostate/mr_test'
    # # Label sources
    # Y_TRAIN_DIR = 'prostate?label_train'
    # Y_VALIDATE_DIR = 'prostate/label_validate'
    # Y_TEST_DIR = 'prostate/label_test'
    #
    # """ Full data & label addresses in Goliath """
    # image_train = sorted([os.path.join(os.getcwd(), 'prostate/mr_train', x)
    #                       for x in os.listdir('prostate/mr_train')])
    # image_validate = sorted([os.path.join(os.getcwd(), 'prostate/mr_validate', x)
    #                          for x in os.listdir('prostate/mr_validate')])
    # image_test = sorted([os.path.join(os.getcwd(), 'prostate/mr_test', x)
    #                      for x in os.listdir('prostate/mr_test')])
    # label_train = sorted([os.path.join(os.getcwd(), 'prostate/label_train', x)
    #                       for x in os.listdir('prostate/label_train')])
    # label_validate = sorted([os.path.join(os.getcwd(), 'prostate/label_validate', x)
    #                          for x in os.listdir('prostate/label_validate')])
    # label_test = sorted([os.path.join(os.getcwd(), 'prostate/label_test', x)
    #                      for x in os.listdir('prostate/label_test')])
    #
    # """ Small test set Goliath"""
    # data_small_train = sorted([os.path.join(os.getcwd(), 'p/data', x)
    #                            for x in os.listdir('p/data')])
    # label_small_train = sorted([os.path.join(os.getcwd(), 'p/label', x)
    #                             for x in os.listdir('p/label')])
    # data_small_validate = sorted([os.path.join(os.getcwd(), 'p/data_validate', x)
    #                               for x in os.listdir('p/data_validate')])
    # label_small_validate = sorted([os.path.join(os.getcwd(), 'p/label_validate', x)
    #                                for x in os.listdir('p/label_validate')])
    # data_small_test = sorted([os.path.join(os.getcwd(), 'p/data_test', x)
    #                           for x in os.listdir('p/data_test')])
    # label_small_test = sorted([os.path.join(os.getcwd(), 'p/label_test', x)
    #                            for x in os.listdir('p/label_test')])

    # """ Test generator, try to visualise - small"""
    # training_generator = sm.ProstateSequence(data_small_train,
    #                                          label_small_train, batch_size=1)
    # validation_generator = sm.ProstateSequence(data_small_validate,
    #                                         label_small_validate, batch_size=1)
    # pred_generator = sm.ProstateSequence(image_test, label_test, batch_size=1, training=False)

    """ GENERATORS"""
    training_generator = sm.ProstateSequence(image_train,
                                             label_train, batch_size=1)
    validation_generator = sm.ProstateSequence(image_validate,
                                               label_validate, batch_size=1)
    pred_generator = sm.ProstateSequence(image_test, label_test, batch_size=1, training=False)


    """ MODEL """
    model = mdl.unet3d(inputsize=(256, 256, 128, 1), kernelSize=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


    # # print model summary
    # with open('model_summary.txt', 'w') as ff:
    #     model.summary(print_fn=lambda x: ff.write(x + '\n'))
    #     # https://newbedev.com/how-to-save-model-summary-to-file-in-keras

    # # Print model structure using pydotplus
    # keras.utils.plot_model(model, "unet3d.png", show_shapes=True)

    # Model fit, plot loss and accuracy
    history = model.fit(training_generator, validation_data=validation_generator, batch_size=1, verbose=2, epochs=3)
    sm.plot_loss(history)
    sm.plot_accuracy(history)
    pred = model.predict(pred_generator)

    # CALCULATE AND PRINT DSC COEF FOR EACH CLASS, AND FOR AVERAGE
    pred_argmax = np.argmax(pred, axis=4)

    # Get an array of test labels -> (18, 256, 256, 128)
    y_true = np.empty((len(label_test), 256, 256, 128, 6))
    for i, id in enumerate(label_test):
        y2 = sm.read_nii(id)
        ohe = tf.keras.utils.to_categorical(y2, num_classes = 6)
        y_true[i,] = ohe
    # print (y_true.shape) # (18,256, 256, 128, 6)

    y_pred_ohe = tf.keras.utils.to_categorical(pred_argmax, num_classes = 6)

    # calculate & print dsc
    dice = sm.dice_coef_multiclass(y_true, y_pred_ohe, 6)


    # PRINT SLICES OF y_true and  y_pred
    sm.slices_pred(y_true, "y_true_bones.png", "y_true_bones") #3
    sm.slices_pred(y_pred_ohe, "y_pred_ohe_bones.png", "y prediction bones") #3


    # PRINT OTHER SLICES
    plt.imshow(pred[0,:,127,:,2])
    plt.title('Prediction Bones')
    plt.savefig('pred.png')
    plt.close()

    plt.imshow(pred_argmax[0,:,127,:])
    plt.title('Prediction argmax')
    plt.savefig('pred_argmax.png')
    plt.close()

    plt.imshow(y_true[0,:,127,:,2])
    plt.title('y_true (to_categorical)')
    plt.savefig('y_true_to_categorical.png')
    plt.close()

    y_true_asis = np.empty((len(label_test), 256, 256, 128))
    for i, id in enumerate(label_test):
        y3 = sm.read_nii(id)
        y_true_asis[i,] = y3
    plt.imshow(y_true_asis[0,:,127,:])
    plt.title('y_true')
    plt.savefig('y_true.png')
    plt.close()


    """ Code to investigate data and images """
    # """ PRINT SLICES OF ONE HOT ENCODED LABEL"""
    # ohe = tf.keras.utils.to_categorical(img_label, num_classes = 6)
    # print(img_label.shape, ohe.shape)
    # print(type(img_label), type(img_mr), type(ohe))
    # sm.slices_ohe(ohe)

    # """ PRINT SLICES OF LABEL"""
    # sm.slices(img_label)

    # """ PRINT SLICES OF DATA """
    # sm.slices(img_mr)

    # #Checks dimensions of each image and label against expected.
    # sm.dim_per_directory()

    # # Display raw data and label info
    # sm.data_info()

    # # Save files for slicing later -> careful, these are very large arrays
    # np.save("y_true", y_true, allow_pickle=False )
    # np.save("y_pred_one", y_pred_ohe, allow_pickle=False )

    # # test print list of label names which include path
    # print(label_test)


if __name__ == '__main__':
    main()
