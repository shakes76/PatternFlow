class Config:
    test_dir = 'dataset/keras_png_slices_test'
    train_dir = 'dataset/keras_png_slices_train'
    val_dir = 'dataset/keras_png_slices_validate'
    BUFFER_SIZE = 4832  # max in 9664
    BATCH_SIZE = 16
    IMG_SIZE = [128, 128]  # .png only
    # data_crop = [0, 0 128, 128] # .jpeg only

    # Training Parameter
    EPOCHS = 30
    NOISE_DIM = 100
    CHECKPOINT_DIR = 'checkpoints/'
    Train = True
    RESTORE = False
