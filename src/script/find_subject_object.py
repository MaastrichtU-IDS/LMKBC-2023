import os
import random
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)


import src.config as config
import util


train_line = util.file_read_json_line(config.TRAIN_FN)
