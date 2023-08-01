import csv
import json
import os
import random
import sys
from regex import cache_all
import requests
from torch import return_types
from tqdm import tqdm
import transformers
import wikipedia
import glob
import re
parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)

import config
import util
import xml.etree.ElementTree as ET
from multiprocessing import Pool, Manager
import numpy as np
from sklearn.model_selection import train_test_split

from datasets import load_dataset,DatasetDict,Dataset,arrow_dataset
import datasets
entity_for_pretrain_fp = f'{config.RES_DIR}/entity_for_pretrain.json'

enhance_tokenizer = transformers.AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
from ahocorapy.keywordtree import KeywordTree



with open(entity_for_pretrain_fp) as f:
    entity_dic_jt = json.load(f)

entity_set = set.union(*[set(e) for e in entity_dic_jt.values()])
print("entity_set",len(entity_set))

kwtree = KeywordTree()
for entity_type, entity_list in entity_dic_jt.items():
    for entity in entity_list:
        if entity_type in {"Autobiography","Series"}:
            kwtree.add(entity)
        else:
            kwtree.add(f" {entity} ")

kwtree.finalize()

entity_dict = dict()
for key in entity_set:
    entity_dict[key] = 0


label= "Country-Language-State"
origin_dir  = f'{config.RES_DIR}/wikidata/origin'

if not os.path.exists( f'{config.RES_DIR}/wikidata/{label}'):
    os.mkdir(f'{config.RES_DIR}/wikidata/{label}')

sentence_dir  = f'{config.RES_DIR}/wikidata/{label}/sentence'
sort_dir  = f'{config.RES_DIR}/wikidata/{label}/sort'
tokenize_dir  = f'{config.RES_DIR}/wikidata/{label}/tokenize'
filter_dir  = f'{config.RES_DIR}/wikidata/{label}/filter'
# filter_dir  = f'{config.RES_DIR}/wikidata/filter'
flatten_sentence_fp = f'{config.RES_DIR}/wikidata/{label}/sentence_leval.json'
filtered_fp = f'{config.RES_DIR}/wikidata/{label}/filter.json'





def data_flatten():
    if os.path.exists(flatten_sentence_fp):
        os.remove(flatten_sentence_fp)
    dataset = Dataset.load_from_disk(sentence_dir)
    record_list=[]
    for row in tqdm(dataset):
        sentence_list = row['sentence']
        entity_list = row['entity']
        for entity, sentence in zip(entity_list,sentence_list):
            # result = kwtree.search_all(sentence)
            # entity_set = set([e for e, s in result])
            if len(entity) == 0:
                continue
            item={
                'sentence':sentence,
                'entities': entity,
                "entity_size":len(entity)
                }    
            # sentence_dataset.add_item(item)
            record_list.append(item)
        if len(record_list) % 1_000_000 == 0:
            util.file_write_json_line(flatten_sentence_fp,record_list,'auto')
            record_list=[]
    util.file_write_json_line(flatten_sentence_fp,record_list,'auto')




    
def split_sentence():
    def mp_shrink_sentence(rows):
        # print("rows", len(rows))
        text_list= rows['text']
        # print("text_list", len(text_list))
        sentence_s = []
        entity_s = []
        new_rows = dict()
        for text  in text_list:
            sentences = text.split('.')
            sentence_list=[]
            entity_list=[]
            for s in sentences:
                # if r'\u' in s:
                #     s = s.replace(r'\u',r'\\u')
                #     s  = s.encode('utf8').decode('unicode-escape')  
                result = kwtree.search_all(s)
                entity_set = set([e.strip() for e, _  in result])
                if len(entity_set) <1: 
                    continue
                sentence_list.append(s)
                entity_list.append(list(entity_set))
            sentence_s.append(sentence_list)
            entity_s.append(entity_list)
        new_rows['sentence']= sentence_s
        new_rows['entity']= entity_s
        return new_rows


    dd = DatasetDict.load_from_disk(origin_dir)
    # dd.cleanup_cache_files()
    dd:Dataset =dd['train']
    # split text into sentences

    dd = dd.map(function = mp_shrink_sentence,
        batched=True,
        batch_size=10000,
        num_proc=15,
        )
    dd.save_to_disk(sentence_dir,
    num_proc=10
    )
    dd.cleanup_cache_files()

def sort_dataset():
    dd =  Dataset.from_json(flatten_sentence_fp)
    dd = dd.sort(  
            column_names="entity_size" ,
            reverse=True
        )
    dd.save_to_disk(sort_dir)
    dd.cleanup_cache_files()

def tokenize_dataset():

    re_multiple_space = re.compile(' +')
    def mp_tokenize(rows):
        entity_list=rows['entities']
        sentence_list=rows['sentence']
        # print(rows['entity_size'])
        token_list = []
        sentence_array = [] 
        for sentence in sentence_list:
            sentence = sentence.replace('\n',' ')
            string_one_space = re_multiple_space.sub(' ', sentence)
            tokens = enhance_tokenizer.tokenize(string_one_space)
            token_list.append(tokens)
            sentence_array.append(string_one_space)
        rows['tokens']=token_list
        rows['sentence'] = sentence_array
        return rows

    dd =  Dataset.load_from_disk(sort_dir)
    dd = dd.map(  mp_tokenize,
        batched=True,
        batch_size=10000,
        num_proc=15,
        )
    dd.save_to_disk(tokenize_dir,
                       num_proc=10)
    dd.cleanup_cache_files()

def filter_dataset():
    def mp_filter_sentence(rows):
        # print("rows", len(rows))
        entity_list=rows['entities']
        sentence_list=rows['sentence']
        token_list = rows['tokens']
        # print(rows['entity_size'])
        result=[]
        for tokens, entities, sentence in zip(token_list, entity_list,sentence_list):
            if len(tokens) > 500:
                result.append(False)
                continue
            entities = set([e for e in entities if e in entity_dict])
            if len(entities) <2:
                continue
            min_count = min([entity_dict[e] for e in entities])

            if  min_count < 50:
                for e in entities:
                    if e in entity_dict:
                        entity_dict[e]+=1
                result.append(True)
            else:
                result.append(False)

        return result


    dd =  Dataset.load_from_disk(tokenize_dir)
    dd = dd.filter(mp_filter_sentence,
            batched=True,
            batch_size=100_000,
            # num_proc=10
            )
    dd.save_to_disk(filter_dir)
    dd.cleanup_cache_files()
    zero_entity_number=0
    for k,v in entity_dict.items():
        if v == 0:
            zero_entity_number +=1 
    print("zero_entity_number",zero_entity_number)

def display_entity_distribution():

    def mp(rows):
        # print("rows", len(rows))
        entity_list=rows['entities']
        sentence_list=rows['sentence']
        token_list = rows['tokens']
        # print(rows['entity_size'])
        # result=[]
        for tokens, entities, sentence in zip(token_list, entity_list,sentence_list):
            entities = set([e for e in entities])
            for e in entities:
                if e in entity_dict:
                    entity_dict[e]+=1 

    entity_type_dict = dict()
    for k, v in entity_dic_jt.items():
        for vi in v:
            entity_type_dict[vi] = k 
    # Q20937
    dd =  Dataset.load_from_disk(filter_dir)
    print("num_rows", dd.num_rows)
    dd = dd.map(mp,
            batched=True,
            batch_size=100_000,
            # num_proc=10
            )
    
    zero_entity_number=0
    entity_collection= list()
    type_count_dict=dict()
    for k,v in entity_dict.items():
        if v == 0:
            zero_entity_number +=1 
        if v<5:
            entity_collection.append(k)
            entity_type = entity_type_dict[k]
            if entity_type not in type_count_dict:
                type_count_dict[entity_type] = 0
            type_count_dict[entity_type]+=1

    # print("type_count_dict", json.dumps(type_count_dict,indent=2))
    print("entity_collection",len(entity_collection))
    for k,v in type_count_dict.items():
        print(k,v)

    type_count_dict.clear()
    print()
    # print("type_count_dict", type_count_dict)
    for k,v in entity_dict.items():
        entity_type = entity_type_dict[k]
        if entity_type not in type_count_dict:
            type_count_dict[entity_type] = 0
        type_count_dict[entity_type]+=v
    print('sentence of each type')
    for k,v in type_count_dict.items():
        print(k,v)
    # for k,v in type_count_dict.items():
    #     print(k, v[:50])
    #     print()
    # print("entity_collection",entity_collection[:100])
    print("zero_entity_number", zero_entity_number)


def export_dataset():
    dd =  Dataset.load_from_disk(filter_dir)

    util.file_delete(filtered_fp)

    dd.to_json(filtered_fp)
    dd.cleanup_cache_files()

def wiki_pipeline():
    print("start split text")
    # split_sentence()
    # flatten sentences into records
    print("start flatten dataset")
    # data_flatten()
    # sort the sentence desc according to entities
    print("start sorting  according entity size")
    # sort_dataset()

    print("start tokenizing sentence ")
    # tokenize_dataset()
    
    # calibrate  
    print("start filtering long sentence ")
    filter_dataset()

    print("start exporting json file ")
    export_dataset()

    display_entity_distribution()


def tree_test():
    s = 'Ko\u0161ice Region'
    result = kwtree.search_all(s)
    print(list(result))

def test_unicode():
    title ="Minamiky\\u016bsh\\u016b"
    normal_title = title.encode('utf8').decode('unicode-escape')
    print(normal_title)

def test_tokenizer():
    tokens = ['0','1','adsfsd']
    ids = enhance_tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    print(enhance_tokenizer.convert_ids_to_tokens(ids))


if __name__ == "__main__":
    wiki_pipeline()
    # tree_test()
    display_entity_distribution()
    # test_unicode()
    # test_tokenizer()

