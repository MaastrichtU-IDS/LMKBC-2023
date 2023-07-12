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


xml_fn = 'res/additional_corpus/simplewiki-20211001-pages-articles-multistream.xml'
tree = ET.parse(xml_fn)
print("parse end")
root = tree.getroot()
for child in root:
    print(child.tag, child.attrib)
