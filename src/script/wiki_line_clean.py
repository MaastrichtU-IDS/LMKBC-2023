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
import xml.etree.ElementTree as ET


xml_fn = f'{config.RES_DIR}/additional_corpus/simplewiki-20211001-pages-articles-multistream.xml'
destination_fn = f"{config.RES_DIR}/additional_corpus/addition_line.txt"
used_fn = f"{config.RES_DIR}/additional_corpus/addition_line_used.txt"
token_fn = f"{config.RES_DIR}/additional_corpus/token_count.txt"
line_count = dict()
line_aim_count = 0
line_need = []
token_path = f"{config.RES_DIR}/tokenizer/bert"
tokenizer = transformers.AutoTokenizer.from_pretrained(token_path)
with open(f'{config.RES_DIR}/tokenizer/bert/added_tokens.json') as f:
    entity_dict = json.load(f)

entity_set_train = entity_dict.keys()
for key in entity_dict.keys():
    entity_dict[key] = 0
with open(destination_fn, 'r') as f:
    f_text = f.read()
    f_lines = f_text.split('\n')
    print(len(f_lines))
    for line in tqdm(f_lines):
        tokens = tokenizer.tokenize(line)
        exists_tokens_count = 0
        exists_tokens = []
        for token in tokens:
            if token in entity_set_train:
                exists_tokens_count += 1
                entity_dict[token] += 1
                exists_tokens.append(token)
        if exists_tokens_count > 1:
            line_need.append((exists_tokens, line))


# line_need.sort(key=lambda x: x[0])
print(len(line_need))
line_used = []
for line in line_need:
    token_str = ','.join(line[0])
    line_str = token_str + '   ' + line[1]
    line_used.append(line_str)

token_count = 0
for key in entity_dict.keys():
    if entity_dict[key] < 1:
        token_count += 1
print('token_count ', token_count)


with open(file=used_fn, mode='w') as f:
    f_text = '\n'.join(line_used)
    f.write(f_text)
with open(file=token_fn, mode='w') as f:
    json.dump(entity_dict, f)
