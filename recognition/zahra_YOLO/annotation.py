#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This script finds the min and max point x and y values where the pixel value is not 0.
#it also calculates the coordinates ratio to the image sizeand generates a txt annotation file for each image.

folder = '/images/ground_truth/ISIC2018_Task1_Training_GroundTruth_x2'
output = '/images/UOLO_DATASET'
for filename in sorted(os.listdir(folder)):
    draw_image = cv2.imread(folder+"/"+filename)
    y_coord_max = max(np.where(draw_image != 0)[0])    
    x_coord_max = max(np.where(draw_image != 0)[1]) 
    y_coord_min = min(np.where(draw_image != 0)[0])    
    x_coord_min = min(np.where(draw_image != 0)[1])

    x_coord_center = (x_coord_min + x_coord_max )/2
    y_coord_center = (y_coord_max + y_coord_min )/2
    image_width = draw_image.shape[1]
    image_height = draw_image.shape[0]
    bounding_box_width = (x_coord_max - x_coord_min)
    bounding_box_heigh = (y_coord_max - y_coord_min)
    x = x_coord_center/image_width
    y = y_coord_center/image_height
    w = bounding_box_width/image_width
    h = bounding_box_heigh/image_height
    cv2.rectangle(draw_image,(x_coord_min, y_coord_min),(x_coord_max,y_coord_max) ,(255, 0, 0), 2)
    new_row = {'object-class':0, 'X': x, 'Y': y, 'width': w, 'height': h}
    df = pd.DataFrame(data=new_row, index=[0])
    df.to_csv(output+'/'+filename[0:-3]+ 'txt',sep=' ',header=False, index=False)
    cv2.imwrite(output+'/'+filename,draw_image)

