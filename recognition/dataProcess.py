
from os import listdir
import shutil
import os

class DataProcess:
  def __init__(self,file_path,right_knee_path,left_knee_path):
    self.file_path = file_path
    self.right_knee_path = right_knee_path
    self.left_knee_path = left_knee_path
    self.data_list = listdir(self.file_path)
    self.data_set_length = len(self.data_list)
  def Process(self):
    r_label_list = ['RIGHT','Right','R_I_G_H_T']
    l_label_list = ['LEFT','Left','L_E_F_T']
    for i in range(self.data_set_length):
      file_name = self.data_list[i]
      if any(label in file_name for label in r_label_list):
        Src = self.file_path + file_name
        Dst = self.right_knee_path + file_name
        shutil.copyfile(Src,Dst)
      if any(label in file_name for label in l_label_list):
        Src = self.file_path + file_name
        Dst = self.left_knee_path + file_name
        shutil.copyfile(Src,Dst)
  def Generate(self, target_path, valid_ratio, test_ratio, class_name):
    assert 0 < valid_ratio < 1
    assert 0 < test_ratio < 1  
    if class_name == 'RIGHT':
        _file_path = self.rightKnee_path
        _data = listdir(self.rightKnee_path)
        _length = len(_data)
    if class_name == 'LEFT':
        _filepath = self.leftKnee_path
        _data = listdir(self.leftKnee_path)
        _length = len(_data)
    trainNum = int(_length * (1 - valid_ratio - test_ratio))
    validNum = int(_length * valid_ratio)
    testNum = int(_length * test_ratio)
    if not os.path.exists(os.path.join(target_path, 'train', class_name)):
            os.makedirs(os.path.join(target_path, 'train', class_name))
    if not os.path.exists(os.path.join(target_path, 'valid', class_name)):
            os.makedirs(os.path.join(target_path, 'valid', class_name))
    if not os.path.exists(os.path.join(target_path, 'test', class_name)):
            os.makedirs(os.path.join(target_path, 'test', class_name))
    for i in range(trainNum):
            src = os.path.join(_file_path, _data[i])
            dst = os.path.join(target_path, 'train', class_name, _data[i])
            shutil.copyfile(src, dst)
    for i in range(trainNum, trainNum + validNum):
            src = os.path.join(_file_path, _data[i])
            dst = os.path.join(target_path, 'valid', class_name, _data[i])
            shutil.copyfile(src, dst)
    for i in range(trainNum + validNum, trainNum + validNum + testNum):
            src = os.path.join(_file_path, _data[i])
            dst = os.path.join(target_path, 'test', class_name, _data[i])
            shutil.copyfile(src, dst)

class_names = ['RIGHT','LEFT']
file_path = '/home/lbd855/DeepLearning/AKOA/'
rightKnee_path = '/home/lbd855/DeepLearning/RIGHT/'
leftKnee_path = '/home/lbd855/DeepLearning/LEFT/'
target_path = '/home/lbd855/DeepLearning/Knee/'
data = DataProcess(file_path, rightKnee_path, leftKnee_path)
data.Process()
for cls in class_names:
    data.Generate(target_path, 0.1, 0.2, cls)
