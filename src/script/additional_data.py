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



xml_fn = f'{config.RES_DIR}/additional_corpus/simplewiki-20211001-pages-articles-multistream.xml'
preprocess_2 = f"{config.RES_DIR}/additional_corpus/preprocess_2.txt"
used_fn = f"{config.RES_DIR}/additional_corpus/py_1.txt"
token_fn = f"{config.RES_DIR}/additional_corpus/token_count.txt"
fm_pretrain_2 = f"{config.RES_DIR}/additional_corpus/fm_pretrain_2.txt"
fm_pretrain_0 = f"{config.RES_DIR}/additional_corpus/fm_pretrain_0.txt"
fm_pretrain_1 = f"{config.RES_DIR}/additional_corpus/fm_pretrain_1.txt"

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


def split_paragraph(text: str):
    lines = []
    words = text.split(' ')
    while len(words) > MAX_LENGTH:
        index = -1
        for i in range(MAX_LENGTH, MIN_LENGTH, -1):
            if words[i] == '.':
                index = i
                break
        if index == -1:
            for i in range(MAX_LENGTH, len(words)):
                if words[i] == '.':
                    words = words[i + 1 :]
                    break
            if i == len(words) - 1:
                break
            else:
                continue
        ws = words[: index + 1]
        lines.append(' '.join(ws))
        words = words[index + 1 :]
    lines.append(' '.join(words))

    return lines

split_sentence_set= {'.',';'}
def split_paragraph_tokens(tokens: list):
    lines = []

    while len(tokens) > MAX_LENGTH:
        index = -1
        for i in range(MAX_LENGTH, MIN_LENGTH, -1):
            if tokens[i] in split_sentence_set:
                index = i
                break
        if index == -1:
            for i in range(MAX_LENGTH, len(tokens)):
                if tokens[i] == '.':
                    tokens = tokens[i + 1 :]
                    break
            if i == len(tokens) - 1:
                break
            else:
                continue
        t_text = tokens[: index + 1]
        lines.append(t_text)
        tokens = tokens[index + 1 :]
    if len(tokens) > MIN_LENGTH:
        lines.append(tokens)
    return lines


def sentence_length(text: str):
    return text.count(' ')


def generate_data_process_fm_ah(origin_data_list):
    global mp_entity_dict
    # line_list = []
    # for line in origin_data_list:
    #     if line.count(' ') > MAX_LENGTH:
    #         line_list.extend(split_paragraph(line))
    #     else:
    #         line_list.append(line)

    result_list = []
    for line in tqdm(origin_data_list):
        if len(line) < 100:
            continue
        line_space_num = float(line.count(' '))
        if line_space_num<20:
            # if line_space_num > 15:
            #     print(line) 
            continue
        results = kwtree.search_all(line)
        results = list(results)
        if results is None or len(results)  <1:
            continue
        entity_count = dict()
        # results  [[a,10],[a,20]]
        all_entity_length = 0
        line_space_num = float(line.count(' '))
        for entity, start in results:
            if entity not in entity_count:
                entity_count[entity] = 0
            entity_count[entity] += len(entity)
            all_entity_length+= len(entity)
        max_count = 0
        for k, v in entity_count.items():
            if v > max_count:
                max_count = v
        token_ratio = (max_count) / line_space_num
        all_token_ratio = all_entity_length/ line_space_num
        if token_ratio > 0.5 or all_token_ratio>0.9:
            continue

        exists_tokens = set((map(lambda x: x[0], results)))
        min_token_count = min(list(map(lambda x: mp_entity_dict[x], exists_tokens)))
        if min_token_count < 1300: 
            for key in exists_tokens:
                mp_entity_dict[key] += 1
            item = {
                "exists": exists_tokens,
                'sentence': line,
            }
            result_list.append(item)

    return result_list


def generate_data_process_nsp_ah(origin_data_list):
    global mp_entity_dict
    line_list = []
    for line in origin_data_list:
        if len(line) > MAX_LENGTH:
            line_list.extend(split_paragraph(line))
        else:
            line_list.append(line)
    result_list = []
    for line in tqdm(line_list):
        results = kwtree.search_all(line)
        results = list(results)
        if results is None or len(results) == 0:
            continue
        exists_tokens = list(map(lambda x: x[0], results))
        min_token_count = min(list(map(lambda x: mp_entity_dict[x], exists_tokens)))
        if len(exists_tokens) > 1 or min_token_count < 20:
            for key in exists_tokens:
                mp_entity_dict[key] += 1
            item = {
                "exists": exists_tokens,
                'sentence': line,
            }
            result_list.append(item)

    return result_list


def dataset_reorder():
    lines: list = util.file_read_json_line(fm_pretrain_0)
    # lines = lines[: len(lines) // 4]
    lines.sort(key=lambda x: len(x['exists']), reverse=True)
    util.file_write_json_line(fm_pretrain_1, lines)


def dataset_optimize():
    dataset_reorder()
    for k in entity_dic_jt.keys():
        entity_dic_jt[k] = 0
    if os.path.exists(fm_pretrain_2):
        os.remove(fm_pretrain_2)
    all_lens = 0
    long_count = 0
    with open(fm_pretrain_1) as f:
        while True:
            lines = f.readlines(1_000_000_000)
            if len(lines) == 0:
                break
            # print(lines[0]
            data_list = []
            for line in lines:
                data_list.append(json.loads(line))
            del lines
            print("data_list ", len(data_list))
            result = []
            for jl in tqdm(data_list):
                exists = set(jl['exists'])
                sentence = jl['sentence']
                sentence=sentence.replace('[[','')
                sentence=sentence.replace(']]','')
                sentence=sentence.replace('|',' ')
                min_count = min(map(lambda x: entity_dic_jt[x], exists))
                if min_count < 100:
                    sentence = jl['sentence'].split('.lt;')[0]
                    sentence = sentence.replace('\n', '')
                    tokens = tokenizer.tokenize(sentence)
                    if len(tokens) < MIN_LENGTH:
                        continue
                    if len(tokens) > MAX_LENGTH:
                        long_count += 1
                        token_list = split_paragraph_tokens(tokens)
                        for toke_sub_list in token_list:
                            if len(toke_sub_list) > MAX_LENGTH:
                                # print("toke_sub_list  ",toke_sub_list)
                                continue
                            token_exists = set()
                            for token in toke_sub_list:
                                if token in mp_entity_dict:
                                    token_exists.add(token)
                            if len(token_exists) < 1:
                                continue
                            min_count = min(                                map(lambda x: entity_dic_jt[x], token_exists)                             )
                            if min_count < 100:
                                item = {
                                    "exists": list(token_exists),
                                    "tokens": toke_sub_list,
                                }
                                for e in token_exists:
                                    entity_dic_jt[e] += 1
                                result.append(item)
                    else:
                        item = {
                            "exists": exists,
                            "tokens": tokens,
                        }
                        for e in exists:
                            entity_dic_jt[e] += 1
                        result.append(item)
            all_lens += len(result)
            print("result", len(result))
            util.file_write_json_line(fm_pretrain_2, result,'auto')
    print("all_lens ", all_lens)
    print("long_count ", long_count)
    k_count = 0
    for k, v in entity_dic_jt.items():
        if v == 0:
            k_count += 1
            # print(k)
    print("unsen_key_count", k_count)


def multi_process_genenrate():
    number_process = 10
    pool = Pool(number_process)
    buffer_size = 500_000_000
    n_size = str(buffer_size / (8 * 1024**3))
    print("batch file size " + n_size + "GB")
    if os.path.exists(fm_pretrain_0):
        os.remove(fm_pretrain_0)
    f_size = os.path.getsize(preprocess_2)
    all_iteration = f_size // buffer_size
    print(f"all_iteration {all_iteration}")
    current_index = 1
    with open(preprocess_2, 'r') as f:
        while True:
            f_lines = f.readlines(buffer_size)
            if len(f_lines) == 0:
                break
            # chunk_size = len(f_text) / (10000000 * number_process)
            chunk_size = 100_000
            print('start process')
            print('batch size', len(f_lines))
            # print("f_lines", f_lines[0])
            chunks = [
                f_lines[i : i + chunk_size] for i in range(0, len(f_lines), chunk_size)
            ]
            # results = [Generate_data_process(f_lines)]
            result_list = pool.map(generate_data_process_fm_ah, chunks)
            util.file_write_json_line(fm_pretrain_0, result_list, 'auto')
            print(f"current/all  {current_index}/{all_iteration}")
            current_index += 1
            # util.file_write_json_line(corpus_3_fn, all_list)
        k_count = 0
        for k, v in mp_entity_dict.items():
            if v == 0:
                k_count += 1
        print("unsen_key_count", k_count)

def test_tokenizer():
    line = 'Fiordland is on the west coast, but is in the Southland, New ZealandSouthland Region rather than the West Coast Region.'
    tokens = tokenizer.tokenize(line)
    print(tokens)


def sample_dev_dataset():
    val_set = util.file_read_json_line('output/filled-mask/filled-mask.jsonl')
    dev_set = random.sample(val_set, 1000)
    dev_set = random.sample(val_set, 1000)
    util.file_write_json_line(config.TRAIN_TINY_FN, dev_set)


def nsp_sample_dev_dataset():
    val_set = util.file_read_json_line('output/filled-mask/filled-mask-train.jsonl')
    random.shuffle(val_set)
    split_index = int(len(val_set) * 0.8)
    train_data = val_set[:split_index]
    test_data = val_set[split_index:]
    util.file_write_json_line("output/filled-mask/train.jsonl", train_data)
    util.file_write_json_line("output/filled-mask/dev.jsonl", test_data)


# def data_analysis():


def test_ah():
    s2 = "82 Template:S.S. Lazio seasons gt; 190506 S.S. Lazio season 190607 S.S. Lazio season 190708 S.S. Lazio season 190809 S.S. Lazio season 190910 S.S. Lazio season 191011 S.S. Lazio season 191112 S.S. Lazio season 191213 S.S. Lazio season 191314 S.S. Lazio season 191415 S.S. Lazio season 191516 S.S. Lazio season 191617 S.S. Lazio season 191718 S.S. Lazio season 191819 S.S. Lazio season 192021 S.S. Lazio season 192122 S.S. Lazio season 192223 S.S. Lazio season 192324 S.S. Lazio season 192425 S.S. Lazio season 192526 S.S. Lazio season 192627 S.S. Lazio season 192728 S.S. Lazio season 192829 S.S. Lazio season 192930 S.S. Lazio season 193031 S.S. Lazio season 193132 S.S. Lazio season 193233 S.S. Lazio season 193334 S.S. Lazio season 193435 S.S. Lazio season 193536 S.S. Lazio season 193637 S.S. Lazio season 193738 S.S. Lazio season 193839 S.S. Lazio season 193940 S.S. Lazio season 194041 S.S. Lazio season 194142 S.S. Lazio season 194243 S.S. Lazio season 194344 S.S. Lazio season 194445 S.S. Lazio season 194546 S.S. Lazio season 194647 S.S. Lazio season 194748 S.S. Lazio season 194849 S.S. Lazio season 194950 S.S. Lazio season 195051 S.S. Lazio season 195152 S.S. Lazio season 195253 S.S. Lazio season 195354 S.S. Lazio season 195455 S.S. Lazio season 195556 S.S. Lazio season 195657 S.S. Lazio season 195758 S.S. Lazio season 195859 S.S. Lazio season 195960 S.S. Lazio season 196061 S.S. Lazio season 196162 S.S. Lazio season 196263 S.S. Lazio season 196364 S.S. Lazio season 196465 S.S."
    rs = kwtree.search_all(s2)
    for r in rs:
        print(s2[r[1] : r[1] + len(r[0])])
    rs = list(rs)
    print(rs)





def clean_sentence(sentence:str):
    s = "In 2005, the SABC announced proposed the creation of two complementary regional television channels, SABC4 and SABC5, to emphasise indigenous languages.lt;refgt;http:allafrica.comstories200506150754.html South Africa: ICASA Grants the Public Broadcaster Licences to Cater for Marginalized Languages, AllAfrica , 15 June 2005lt;refgt; SABC4, based in Mafikeng, was to be broadcast in Tswana languageTswana , Sesotho languageSesotho , Pedi languagePedi , Tsonga languageTsonga , Venda languageVenda , and Afrikaans, to the northern provinces of the country, while SABC5, based in Cape Town, was to broadcast in Xhosa languageXhosa , Zulu languageZulu , Southern Ndebele languageNdebele , and Swazi languageSwazi , as well as Afrikaans, to the southern provinces. Unlike other SABC TV services, SABC4 and SABC5 were not to be available via satellite.lt;refgt;http:www.news24.comSouthAfricaPoliticsSABCsreadytoroll20050314 SABCs ready to roll, News24 websiteNews24 , 14 March 2005lt;refgt; Apart from soundbites on news or current affairs programmes, no Englishlanguage programming would be shown on either channel.lt;refgt;https:variety.com2005scenemarketsfestivalssabcaddschannels1117924696 SABC adds channels, Variety magazineVariety , 19 June 2005lt;refgt; However, the plans fell through and in 2015, the SABC stated that it would launch two new channels, SABC News and SABC Encore.lt;refgt;http:www.channel24.co.zaTVNewsHlaudiSABCwillnowstartDTTwith5TVchannels20150520 Hlaudi: SABC will now start DTT with 5 TV channels, News24 websiteNews24 , 20 May 2015lt;refgt;\n"

def check_row(row):
    result = kwtree.search(row['text'])
    return result is not None 

def split_text(row):
    sentences = row['text'].split('.')
    sentence_list = []
    for sentence in sentences:
        # if len(sentence) <50:
        #     continue
        result = kwtree.search(sentence)
        if result is not None: 
            sentence_list.append( sentence)
    row['text'] = sentence_list
    # return 

def length_check(row):
    # result = list(kwtree.search_all(row['text']))
    return len(row['text']) > 30



def wiki_sentence_count():
    wiki_splitted = f'{config.RES_DIR}/wikidata/splited'
    # dataset.save_to_disk(wiki_origin_dir)
    dataset = DatasetDict.load_from_disk(wiki_splitted)
    sentence_count = 0
    # dataset = dataset.map(function = lambda x : sentence_count=sentence_count+len(x['text']), 
    #                         batched=True,
    #                     #  batch_size=1000
    #                         num_proc=5
    #                         )
    print("sentence_count", sentence_count)

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
            if  min_count < 50:
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