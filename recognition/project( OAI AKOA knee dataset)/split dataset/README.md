# Contents
* ## Description
     This part is to classify the raw dataset as two classe(left knee and right knee). This is to sort the images into two different folders based on the keyword of the file name (keyword:'RIGHT'). Then split them into training dataset (12,000 images), validation dataset (4,000 images) and test dataset(2,680 images).

* ## Results 
   Images classification:
   
  ![](https://github.com/1665446266/PatternFlow/blob/topic-recognition/recognition/project(%20OAI%20AKOA%20knee%20dataset)/split%20dataset/images%20classification.png?raw=true)
   
   Train,validation and test dataset:
   
  ![](https://github.com/1665446266/PatternFlow/blob/topic-recognition/recognition/project(%20OAI%20AKOA%20knee%20dataset)/split%20dataset/train,validation%20and%20test%20dataset.png?raw=true)
  
* ## Code

   ```python
   __author__ = 'xiaomeng cai'
   __date__ = '10/14/2020 '

   import shutil
   import os
   import re

   path = r'E:\COMP3710DATA\AKOA_Analysis'
   files_list = os.listdir(path)

   left_path = r"E:\COMP3710DATA\left"
   right_path = r"E:\COMP3710DATA\right"

   # split the dataset into 2 classes(right, left).
   for file in files_list:
       file_name = re.split('[_.]',file)
       if "RIGHT" in file_name:
           shutil.move(os.path.join(path,file),os.path.join(right_path,file))
       else:
           shutil.move(os.path.join(path,file),os.path.join(left_path,file))

   # split the dataset into train, validation and test dataset
   base_path = r'E:\COMP3710DATA\dataset'
   
   # creat folder
   train_dir = os.path.join(base_path, 'train')
   os.mkdir(train_dir)
   validation_dir = os.path.join(base_path, 'validation')
   os.mkdir(validation_dir)
   test_dir = os.path.join(base_path, 'test')
   os.mkdir(test_dir)

   train_right_dir = os.path.join(train_dir, 'right')
   os.mkdir(train_right_dir)
   train_left_dir = os.path.join(train_dir, 'left')
   os.mkdir(train_left_dir)

   validation_right_dir = os.path.join(validation_dir, 'right')
   os.mkdir(validation_right_dir)
   validation_left_dir = os.path.join(validation_dir, 'left')
   os.mkdir(validation_left_dir)

   test_right_dir = os.path.join(test_dir, 'right')
   os.mkdir(test_right_dir)
   test_left_dir = os.path.join(test_dir, 'left')
   os.mkdir(test_left_dir)
   
   # py the images
   def copy_images (path1,path2,num1,num2):
       for file in os.listdir(path1)[num1:num2]:
           src = os.path.join(path1, file)
           dst = os.path.join(path2, file)
           shutil.copyfile(src, dst)

   copy_images(left_path,train_left_dir,0,6000)
   copy_images(left_path,validation_left_dir,6000,8000)
   copy_images(left_path,test_left_dir,8000,9040)

   copy_images(right_path,train_right_dir,0,6000)
   copy_images(right_path,validation_right_dir,6000,8000)
   copy_images(right_path,test_right_dir,8000,9640)

   ```
