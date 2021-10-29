#This script reads all the images and resizes the image without changing the aspect ratio.
#in this script we adjust the size of the image by adding padding to the top, bottom, left, and right of the image

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.read().splitlines() #file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 1

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path)
        height , width = img.size
        img_size=416  
        ratio = min(img_size/img.size[0], img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        
        #this is the padding for the left, top, right and bottom borders respectively.
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
        image_tensor = img_transforms(img).float()
    # convert image to Tensor
        input_img = img_transforms(img).float()
        
        
# this script extract the coordinates of the bounding box we created in the annotation file for the original image which is unpadded
#and unscaled, then updates the left-bottom and top-right coordinate of the bounding box according to the scaled and padded image

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        pad = ((pad1, pad2), (0, 0), (0, 0)) if height <= width else ((0, 0), (pad1, pad2), (0, 0))
        padded_height, padded_width, _ = input_img.shape
        
        annotation = None
        if os.path.exists(label_path):
            annotation = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = width * (annotation[:, 1] - annotation[:, 3]/2)
            y1 = height * (annotation[:, 2] - annotation[:, 4]/2)
            x2 = width * (annotation[:, 1] + annotation[:, 3]/2)
            y2 = height * (annotation[:, 2] + annotation[:, 4]/2)
            
            
            
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            annotation[:, 1] = ((x1 + x2) / 2) / padded_width
            annotation[:, 2] = ((y1 + y2) / 2) / padded_height
            annotation[:, 3] *= width / padded_width
            annotation[:, 4] *= height / padded_height
        
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if annotation is not None:
            filled_labels[range(len(annotation))[:self.max_objects]] = annotation[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)