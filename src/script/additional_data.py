import csv
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
from multiprocessing import Pool, Manager
import numpy as np
from sklearn.model_selection import train_test_split


xml_fn = f'{config.RES_DIR}/additional_corpus/simplewiki-20211001-pages-articles-multistream.xml'
contain_words = f"{config.RES_DIR}/additional_corpus/contain_words.txt"
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
entity_dict = manager.dict()
with open(f'{config.RES_DIR}/tokenizer/bert/added_tokens.json') as f:
    entity_dic_jt = json.load(f)

entity_set_train = entity_dic_jt.keys()
for key in entity_dic_jt.keys():
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
    global entity_dict
    # line_list = []
    # for line in origin_data_list:
    #     if line.count(' ') > MAX_LENGTH:
    #         line_list.extend(split_paragraph(line))
    #     else:
    #         line_list.append(line)

    result_list = []
    for line in tqdm(origin_data_list):
        # if len(line) < 100:
        #     continue
        line_space_num = float(line.count(' '))
        if line_space_num<20:
            if line_space_num > 15:
                print(line) 
            continue
        results = kwtree.search_all(line)
        results = list(results)
        if results is None or len(results) == 0:
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
        min_token_count = min(list(map(lambda x: entity_dict[x], exists_tokens)))
        if min_token_count < 300: 
            for key in exists_tokens:
                entity_dict[key] += 1
            item = {
                "exists": exists_tokens,
                'sentence': line,
            }
            result_list.append(item)

    return result_list


def generate_data_process_nsp_ah(origin_data_list):
    global entity_dict
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
        min_token_count = min(list(map(lambda x: entity_dict[x], exists_tokens)))
        if len(exists_tokens) > 1 or min_token_count < 20:
            for key in exists_tokens:
                entity_dict[key] += 1
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

                min_count = min(map(lambda x: entity_dic_jt[x], exists))
                if min_count < 50:
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
                                if token in entity_dict:
                                    token_exists.add(token)
                            if len(token_exists) < 1:
                                continue
                            min_count = min(                                map(lambda x: entity_dic_jt[x], token_exists)                             )
                            if min_count < 50:
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
    f_size = os.path.getsize(contain_words)
    all_iteration = f_size // buffer_size
    print(f"all_iteration {all_iteration}")
    current_index = 1
    with open(contain_words, 'r') as f:
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
        for k, v in entity_dict.items():
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


def official_language():
    official_language_fn = 'res/additional_corpus/official_language.txt'
    with open(official_language_fn) as f:
        csv_dict = csv.DictReader(f)
        result=[]
        for row in csv_dict: 
            country = row['Country'].strip().replace('&nbsp;','')
            language = row['Official'].strip().split(' ')
            result .append((country,language))
    print(result)



if __name__ == "__main__":
    # test_ah()
    # multi_process_genenrate()
    # official_language()
    # clean_file(used_fn)
    dataset_optimize()
    # sample_dev_dataset()
    # nsp_sample_dev_dataset()
    # test_tokenizer()
    # dataset_reorder()
