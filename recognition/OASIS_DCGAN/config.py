class Config:
    TEST_DIR = 'dataset/keras_png_slices_test'
    TRAIN_DIR = 'dataset/keras_png_slices_train'
    VALID_DIR = 'dataset/keras_png_slices_validate'
    BUFFER_SIZE = 5000  # maximum in 9664
    BATCH_SIZE = 16
    IMG_SIZE = [128, 128]  # .png only
    # data_crop = [0, 0 128, 128] # .jpeg only

    # Training Parameter
    EPOCHS = 40
    NOISE_DIM = 100
    CHECKPOINT_DIR = 'checkpoints/'

