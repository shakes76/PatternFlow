import git
import os

def get_yolo():
  current_dir = os.getcwd()
  git.Git(current_dir).clone('https://github.com/ultralytics/yolov5.git')


get_yolo()