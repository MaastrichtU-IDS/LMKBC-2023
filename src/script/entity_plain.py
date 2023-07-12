import json
import os
import random
import sys
import requests
from tqdm import tqdm
import transformers
import wikipedia


parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)

import config
import util


entity_fn = f"{config.RES_DIR}/additional_corpus/entity.txt"

with open(f'{config.RES_DIR}/tokenizer/bert/added_tokens.json') as f:
    entity_dict = json.load(f)

with open(entity_fn, 'w') as f:
    lines = '\n'.join(entity_dict.keys())
    f.write(lines)
