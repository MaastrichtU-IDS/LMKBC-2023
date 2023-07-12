import gc
import json
import os
import random
import sys
import time
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
from tqdm import tqdm

xml_fn = f'{config.RES_DIR}/additional_corpus/simplewiki-20211001-pages-articles-multistream.xml'
xml_large = (
    f"{config.RES_DIR}/additional_corpus/enwiki-20230401-pages-articles-multistream.xml"
)
destination_fn = f"{config.RES_DIR}/additional_corpus/addition_line.txt"
MAX_LENGTH = 1500
MIN_LENGTH = 100


def split_paragraph(text: str):
    lines = []

    while len(text) > MAX_LENGTH:
        index = text.rfind(
            '. ',
            100,
            500,
        )
        if index == -1:
            index = MAX_LENGTH
        t_text = text[: index + 1]
        lines.append(t_text)
        text = text[index + 1 :]
    lines.append(text)

    return lines


def process(f_lines):
    line_need = []
    line_aim_count = 0
    print("f_lines", len(f_lines))
    for line in tqdm(f_lines):
        line_len = len(line)
        if line.find("</") != -1:
            continue
        if line_len > MIN_LENGTH:
            if line_len < MAX_LENGTH:
                line_aim_count += 1
                line_need.append(line)
            else:
                splited_lines = split_paragraph(line)
                line_need.extend(splited_lines)
    print("line_aim_count", line_aim_count)
    print("lines", len(line_need))
    if os.path.exists(destination_fn):
        m = 'a'
    else:
        m = 'w'
    with open(destination_fn, m) as f:
        f_text = '\n'.join(line_need)
        f.write(f_text)
    del f_text
    del f_lines
    del line_need


with open(xml_large, 'r') as f:
    f_text = f.read(1_000_000_000)
    while f_text:
        f_lines = f_text.split('\n')
        process(f_lines)
        gc.collect()
        time.sleep(30)
        f_text = f.read(1_000_000_000)
