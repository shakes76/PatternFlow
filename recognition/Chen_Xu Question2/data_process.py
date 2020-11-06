gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)


from pathlib import Path
from zipfile import ZipFile
import tensorflow as tf

def load_localData(path = '/content/drive/My Drive/colab/P4/AKOA_Analysis.zip'):
  # path = '/content/drive/My Drive/colab/P4/AKOA_Analysis.zip'
  raw_file = ZipFile(path)
  raw_list = raw_file.namelist()
  right_list = filter(lambda x: 'right' in x.lower() or 'r_i_g_h_t' in x.lower() ,raw_list)
  left_list = filter(lambda x: 'left' in x.lower() or 'l_e_f_t' in x.lower() ,raw_list)

  for vowel in right_list:
    raw_file.extract(vowel, 'data/right')
  for vowel in left_list:
    raw_file.extract(vowel, 'data/left')
  
load_localData()
