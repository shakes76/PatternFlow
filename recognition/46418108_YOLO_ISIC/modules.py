import git
import os

def get_yolo():
  """
  This function clones the YOLOv5 repositry to current working directory.
  The user will need to have Git setup as well as the python Git package
  installed. 
  """
  current_dir = os.getcwd()
  git.Git(current_dir).clone('https://github.com/ultralytics/yolov5.git')

