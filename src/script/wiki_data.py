


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

token_path = f"{config.RES_DIR}/tokenizer/bert"
tokenizer = transformers.AutoTokenizer.from_pretrained(token_path)

MAX_LENGTH = 500
MIN_LENGTH = 10

from ahocorapy.keywordtree import KeywordTree

kwtree = KeywordTree()

manager = Manager()
mp_entity_dict = manager.dict()
with open(f'{config.RES_DIR}/tokenizer/bert/added_tokens.json') as f:
    entity_dic_jt = json.load(f)
entity_dict = dict()
entity_set_train = entity_dic_jt.keys()
for key in entity_dic_jt.keys():
    mp_entity_dict[key] = 0
    entity_dict[key] = 0
    kwtree.add(key)

kwtree.finalize()

wiki_origin_dir  = f'{config.RES_DIR}/wikidata/origin'
wiki_shrink = f'{config.RES_DIR}/wikidata/shrink'
wiki_shrink_0 = f'{config.RES_DIR}/wikidata/shrink_0'
wiki_shrink_1 = f'{config.RES_DIR}/wikidata/shrink_1'
wiki_sentence_leval = f'{config.RES_DIR}/wikidata/sentence_leval'
wiki_sentence_leval_json = f'{config.RES_DIR}/wikidata/sentence_leval,json'
wiki_sentence_sorted = f'{config.RES_DIR}/wikidata/sentence_sorted'
wiki_token_size = f'{config.RES_DIR}/wikidata/token_size'
wiki_sentence_filter_entity_size = f'{config.RES_DIR}/wikidata/sentence_filter_entity_size'
wiki_sentence_filter_entity_size_json = f'{config.RES_DIR}/wikidata/sentence_filter_entity_size.json'

def mp_shrink_sentence(rows):
    # print("rows", len(rows))
    text_list= rows['sentence']
    # print("text_list", len(text_list))
    sentence_s = []
    entity_s = []
    new_rows = dict()
    for i in range(len(text_list)):
        text = text_list[i]
        sentences = text.split('.')
        sentence_list=[]
        entity_list=[]
        for s in sentences:
            result = kwtree.search_all(s)
            entity_set = set([e for e, s in result])
            if len(entity_set) == 0: 
                continue
            sentence_list.append(s)
            entity_list.append(list(entity_set))
        sentence_s.append(sentence_list)
        entity_s.append(entity_list)
    new_rows['sentence']= sentence_s
    new_rows['entity']= entity_s
    return new_rows


def shrink_sentence(rows):
    # print("rows", len(rows))
    text_list= rows['sentence']
    entity_list=rows['sentence']
    # print("text_list", len(text_list))
    sentence_s = []
    entity_s = []
    new_rows = dict()
    for i in range(len(text_list)):
        text = text_list[i]
        # sentences = text.split('.')
        sentence_list=[]
        entity_list
        for s in sentences:
            result = kwtree.search_all(s)
            entity_set = set([e for e, s in result])
            if len(entity_set) == 0:
                continue
            min_count = min([entity_dict[e] for e in entity_set])
            if result is not None and min_count < 50:
                for e in entity_set:
                    entity_dict[e]+=1
                sentence_list.append(s)
        sentence_s.append(sentence_list)
    new_rows['sentence']= sentence_s
    return new_rows


def save_orivin():
    dd = DatasetDict.load_from_disk(wiki_origin_dir)
    dd.save_to_disk(wiki_shrink)

def mp_shrink_text(rows):
    # print("rows", len(rows))
    text_list= rows['text']
    # print("text_list", len(text_list))
    sentence_s = []
    for i in range(len(text_list)):
        text = text_list[i]
        sentences = text.split('.')
        sentence_list=[]
        for s in sentences:
            result = kwtree.search(s)
            if result is not None:
                sentence_list.append(s)
        sentence_s.append(sentence_list)
    rows['sentence']= sentence_s
    del rows['text']
    return rows


def wiki_filter():
    dd = DatasetDict.load_from_disk(wiki_shrink_0)
    print(dd.column_names)

def count_entity(rows):
    # print("rows", len(rows))
    text_list= rows['sentence']
    # print("text_list", len(text_list))
    sentence_s = []
    new_rows = dict()
    for i in range(len(text_list)):
        sentences = text_list[i]
        # print("text",len(text))
        # sentences = text.split('.')
        sentence_list=[]
        for s in sentences:
            result = kwtree.search_all(s)
            entity_set = set([e for e, s in result])
            if len(entity_set) == 0:
                continue
            for e in entity_set:
                entity_dict[e]+=1
                # sentence_list.append(s)
            # min_count = min([entity_dict[e] for e in entity_set])
            # if result is not None and min_count < 50:
            #     for e in entity_set:
            #         entity_dict[e]+=1
            #     sentence_list.append(s)
        # sentence_s.append(sentence_list)
    # new_rows['sentence']= sentence_s
    # return new_rows

def display_zero_entity():
    dd = DatasetDict.load_from_disk(wiki_shrink_1)
    dd1 = dd.map(function = count_entity,
           batched=True,
           batch_size=1000,
        #    num_proc=15,
           )
    zero_entity_number=0
    for k,v in entity_dict.items():
        if v == 0:
            zero_entity_number +=1 
    print("zero_entity_number",zero_entity_number)

def shrink_file():
    dd = DatasetDict.load_from_disk(wiki_shrink_0)
    dd1 = dd.map(function = mp_shrink_sentence,
           batched=True,
           batch_size=1000,
           num_proc=15,
           )
    # zero_entity_number = 
    # dd1.set_format()()
    dd1.save_to_disk(wiki_shrink_1)

def wiki_data_display():
    dd = DatasetDict.load_from_disk(wiki_sentence_leval)
    ds=dd['train']
    print(ds[1000]['sentnece'])
    print(ds[1000]['entities'])

    dataset = DatasetDict.load_from_disk(wiki_sentence_filter)['train']
    print(dataset[1000]['text'])
    print(dataset[1000]['entity'])




def wiki_filter_entity_size_single():
    dataset = Dataset.load_from_disk(wiki_sentence_sorted)


def wiki_filter_entity_size_simple():
 
    dataset = util.file_read_json_line(wiki_token_size)
    for row in dataset:
        entities=row['entities']
        sentence=row['sentence']
        tokens= row['tokens']
        min_count = min([entity_dict[e] for e in entities])
        if min_count > 50:
            continue
        tokens = tokenizer.tokenize(sentence)
        if len(tokens>500):
            continue
        for e in entities:
            entity_dict[e]+=1

    zero_entity_number=0
    for k,v in entity_dict.items():
        if v == 0:
            zero_entity_number +=1 
    print("zero_entity_number",zero_entity_number)
    # dataset.save_to_disk(wiki_sentence_filter_entity_size)
    util.file_delete(wiki_sentence_filter_entity_size_json)
    dataset.to_json(wiki_sentence_filter_entity_size_json)



def wiki_filter_token_length():

    def map_func(rows):
        entity_list=rows['entities']
        sentence_list=rows['sentence']
        # print(rows['entity_size'])
        token_list = []
        sentences = [] 
        for sentence in sentence_list:
            sentence = sentence.replace('\n','')
            tokens = tokenizer.tokenize(sentence)
            token_list.append(tokens)
            sentences.append(sentence)
        rows['tokens']=token_list
        rows['sentence'] = sentences
        return rows
    
    def filter_func(rows):
        # print("rows", len(rows))
        # entity_list=rows['entities']
        sentence_list=rows['sentence']
        token_list = rows['tokens']
        # print(rows['entity_size'])
        result=[]
        for tokens  in token_list:
            if  len(tokens) < 500:
                result.append(True)
            else:
                result.append(False)
        return result

    dataset = Dataset.load_from_disk(wiki_sentence_sorted)
    print(dataset.num_rows)
    print(dataset.column_names)
    dataset = dataset.map(map_func,
                batched=True,
                batch_size=100_000,
                num_proc=10
                )
    dataset = dataset.filter(filter_func,
                batched=True,
                batch_size=100_000,
                num_proc=10
                )
    print(dataset.num_rows)
    print(dataset.column_names)
    dataset.save_to_disk(wiki_token_size,
                         num_proc=10)


def wiki_filter_entity_size():
    def wiki_filter_sentence(rows):
        # print("rows", len(rows))
        entity_list=rows['entities']
        sentence_list=rows['sentence']
        # print(rows['entity_size'])
        result=[]
        for entities, sentence in zip(entity_list,sentence_list):
            min_count = min([entity_dict[e] for e in entities])
            if  min_count < 20:
                for e in entities:
                    entity_dict[e]+=1
                result.append(True)
            else:
                result.append(False)

        return result


    dataset = Dataset.load_from_disk(wiki_token_size)
    print(dataset.num_rows)
    print(dataset.column_names)
    dataset = dataset.filter(wiki_filter_sentence,
                batched=True,
                batch_size=100_000,
                # num_proc=10
                )
    print(dataset.num_rows)
    print(dataset.column_names)
    zero_entity_number=0
    for k,v in entity_dict.items():
        if v == 0:
            zero_entity_number +=1 
    print("zero_entity_number",zero_entity_number)
    # dataset.save_to_disk(wiki_sentence_filter_entity_size)
    util.file_delete(wiki_sentence_filter_entity_size_json)
    dataset.to_json(wiki_sentence_filter_entity_size_json)
  
def wiki_sort():
    dataset = Dataset.from_json(wiki_sentence_leval_json)
    dataset = dataset.sort(  
        column_names="entity_size" ,
                 reverse=True
               )
    dataset.save_to_disk(wiki_sentence_sorted)

def wiki_data_flatten():
    dataset = DatasetDict.load_from_disk(wiki_shrink_1)
    if os.path.exists(wiki_sentence_leval_json):
        os.remove(wiki_sentence_leval_json)
    record_list=[]
    for row in tqdm(dataset['train']):
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
        if len(record_list) % 10_000_000 == 0:
            util.file_write_json_line(wiki_sentence_leval_json,record_list,'auto')
            record_list=[]
    util.file_write_json_line(wiki_sentence_leval_json,record_list,'auto')
            # sentence_dataset.flatten_indices(r)
    # ds = Dataset.from_list(record_list)
    # sentence_dataset['train']= sentence_dataset
    # sentence_dataset.to_(wiki_sentence_leval)


def wiki_data():
    dataset = DatasetDict.load_from_disk(wiki_filter)
    print(dataset.column_names)
    print(dataset.num_rows)
    dataset = dataset.map(function = sentence_check, 
                             batched=True,
                             batch_size=100,
                             num_proc=10,
                            # return_dict=False 
                             )
    
    # dataset = dataset.filter(function = check_row, 
    #                         #  batched=True,
    #                         #  batch_size=1000
    #                         num_proc=10
    #                          )
    print(dataset.num_rows)
    print(dataset['train'][0])

    dataset.save_to_disk(wiki_sentence_filter)
    # print(dataset["train"]["text"][:10])
    # print(dataset)

def sentence_check(rows):

    text_list = rows['text']
    if len(text_list) == 0:
        print(" empty")
        return
    rows['entity'] = []
    for i, text in enumerate(text_list):
        # if len(sentence) <50:
        #     continue
        # text = text[text]
        sentence_list = []
        entity_list=[]
        sentences = text.split('.')
        for sentence in sentences:
            result = kwtree.search_all(sentence)
            if result is not None: 
                # if len(list(result)) > 30:
                sentence_list.append( text)
                entity_list.append([e for e,s in result])
                # entity_set = set([e for e,s in result])
                # min_num = 100
                # for e in entity_set:
                #     if entity_dict[e] < min_num:
                #         min_num = entity_dict[e]
                # if min_num <100:
                #     for e in entity_set:
                #         entity_dict[e] +=1
                #     sentence_list.append( sentence)
        text_list[i]= sentence_list
        rows['entity'].append(sentence_list)
    return rows

def wiki_mp(fp):
    basename = os.path.basename (fp)
    # dataset = arrow_dataset.ArrowReader(fp)
    # dataset.read_table()
    dataset = dataset.map(function = sentence_check)
    dataset.save_to_disk(f'{config.RES_DIR}/wikidata/splited_mp/{basename}')


def wiki_multi_process():
    number_process = 10
    pool = Pool(number_process)
    wiki_origin_dir  = f'{config.RES_DIR}/wikidata/splited/train'
    files = glob.glob (f'{wiki_origin_dir}/data*.arrow')
    print (files)
    pool.map(wiki_mp,files)



if __name__ == "__main__":
    # test_ah()
    # multi_process_genenrate()
    # official_language()
    # clean_file(used_fn)
    # dataset_optimize()
    # sample_dev_dataset()
    # nsp_sample_dev_dataset()
    # test_tokenizer()
    # dataset_reorder()
    # wiki_data()
    # wiki_multi_process()
    # wiki_sentence_count()
    # wiki_data_filter()
    # wiki_data_display()
    # save_orivin()
    # shrink_file()
    # wiki_filter()
    # display_zero_entity()
    # wiki_data_flatten()
    # wiki_sort()
    wiki_filter_entity_size()
    # wiki_filter_token_length()
