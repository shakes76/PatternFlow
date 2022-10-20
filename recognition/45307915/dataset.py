import tensorflow as tf

TRAIN_IMAGES_PATH = "./ISIC-2017_Training_Data/*.jpg"
TRAIN_MASKS_PATH = "./ISIC-2017_Training_Part1_GroundTruth/*.png"

TEST_IMAGES_PATH = "./ISIC-2017_Test_v2_Data/*.jpg"
TEST_MASKS_PATH = "./ISIC-2017_Test_v2_Part1_GroundTruth/*.png"

VALIDATE_IMAGES_PATH = "./ISIC-2017_Validation_Data/*.jpg"
VALIDATE_MASKS_PATH = "./ISIC-2017_Validation_Part1_GroundTruth/*.png"

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

class DataLoader():
    
    def __init__(self, train_images_path=TRAIN_IMAGES_PATH, train_masks_path=TRAIN_MASKS_PATH, 
                 test_images_path=TEST_IMAGES_PATH, test_masks_path=TEST_MASKS_PATH,
                 validate_images_path=VALIDATE_IMAGES_PATH, validate_masks_path=VALIDATE_MASKS_PATH, 
                 img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
        """ Create a new Yolo Model instance.
        
        Parameters:
            images_path (str): Path of the dataset images
            masks_path (str):  Path of the dataset masks
            img_width (int): Image Width
            img_height (int): Image Height
        """
        self.train_images_path = train_images_path
        self.train_masks_path = train_masks_path
        
        self.test_images_path = test_images_path
        self.test_masks_path = test_masks_path
        
        self.validate_images_path = validate_images_path
        self.validate_masks_path = validate_masks_path
        
        self.img_width = img_width
        self.img_height = img_height
        
    def preprocessImages(self, filenames):
        """
        Load and preprocess the image files.

        Parameters:
            filenames (tf.string): names of all image files

        Return:
            tf.Dataset: A (img_height, img_width, 1) tensor containing all the image file data

        """
        raw = tf.io.read_file(filenames)
        images = tf.io.decode_jpeg(raw, channels=1)

        #resize images
        images = tf.image.resize(images, [self.img_height, self.img_width])

        #Normalise
        images = images / 255.

        return images


    def preprocessMasks(self, filenames):
        """
        Load and preprocess the mask files.

        Parameters:
            filenames (tf.string): names of all mask files

        Return:
            tf.Dataset: A (img_height, img_width, 1) tensor containing all the mask file data

        """
        raw = tf.io.read_file(filenames)
        images = tf.io.decode_png(raw, channels=1)

        #resize images
        images = tf.image.resize(images, [self.img_height, self.img_width])

        #Normalise
        images = images / 255.

        #Threshold image to 0-1
        images = tf.where(images > 0.5, 1.0, 0.0)

        return images
    
    def loadData(self):
        """
        Loads and prepocesses all the image and mask data, 
        for the training, testing and validation datasets


        Return:
            (tf.Dataset, tf.Dataset, tf.Dataset): 3 (img_height, img_width, 1) tensors 
                containing all the image and mask data
        """
        
        #train
        train_image_data = tf.data.Dataset.list_files(self.train_images_path, shuffle=False)
        train_processed_images = train_image_data.map(self.preprocessImages)

        train_mask_data = tf.data.Dataset.list_files(self.train_masks_path, shuffle=False)
        train_processed_masks = train_mask_data.map(self.preprocessMasks)

        train_dataset = tf.data.Dataset.zip((train_processed_images, train_processed_masks))
        
        #test
        test_image_data = tf.data.Dataset.list_files(self.test_images_path, shuffle=False)
        test_processed_images = test_image_data.map(self.preprocessImages)

        test_mask_data = tf.data.Dataset.list_files(self.test_masks_path, shuffle=False)
        test_processed_masks = test_mask_data.map(self.preprocessMasks)

        test_dataset = tf.data.Dataset.zip((test_processed_images, test_processed_masks))
        
        #validate
        validate_image_data = tf.data.Dataset.list_files(self.validate_images_path, shuffle=False)
        validate_processed_images = validate_image_data.map(self.preprocessImages)

        validate_mask_data = tf.data.Dataset.list_files(self.validate_masks_path, shuffle=False)
        validate_processed_masks = validate_mask_data.map(self.preprocessMasks)

        validate_dataset = tf.data.Dataset.zip((validate_processed_images, validate_processed_masks))
        
        return train_dataset, test_dataset, validate_dataset