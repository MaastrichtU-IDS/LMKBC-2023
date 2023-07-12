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

xml_fn = f'{config.RES_DIR}/additional_corpus/simplewiki-20211001-pages-articles-multistream.xml'
destination_fn = f"{config.RES_DIR}/additional_corpus/contain_words.txt"
used_fn = f"{config.RES_DIR}/additional_corpus/py_1.txt"
token_fn = f"{config.RES_DIR}/additional_corpus/token_count.txt"
final_corpus_fn = f"{config.RES_DIR}/additional_corpus/final_corpus.txt"
corpus_json_fn = f"{config.RES_DIR}/additional_corpus/corpus_json.txt"
corpus_json_order_fn = f"{config.RES_DIR}/additional_corpus/corpus_json_order.txt"

token_path = f"{config.RES_DIR}/tokenizer/bert"
tokenizer = transformers.AutoTokenizer.from_pretrained(token_path)

MAX_LENGTH = 1500
MIN_LENGTH = 100

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

    while len(text) > MAX_LENGTH:
        index = text.rfind(
            '. ',
            MIN_LENGTH,
            MAX_LENGTH,
        )
        if index == -1:
            index = MAX_LENGTH
        t_text = text[: index + 1]
        lines.append(t_text)
        text = text[index + 1 :]
    lines.append(text)

    return lines


def split_paragraph_tokens(tokens: list):
    lines = []

    while len(tokens) > MAX_LENGTH:
        index = -1
        for i in range(MAX_LENGTH, MIN_LENGTH, -1):
            if tokens[i] == '.':
                index = i
                break
        if index == -1:
            index = MAX_LENGTH
        t_text = tokens[: index + 1]
        lines.append(t_text)
        tokens = tokens[index + 1 :]
    if len(tokens) > MIN_LENGTH:
        lines.append(tokens)
    return lines


def token_analysis():
    with open(token_fn) as f:
        entity_dict = json.load(f)
    k_count = 0
    for k, v in entity_dict.items():
        if v == 0:
            k_count += 1
            print(k)

    print("k_count", k_count)


def generate_data_process_ah(origin_data_list):
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
    lines: list = util.file_read_json_line(corpus_json_fn)
    # lines = lines[: len(lines) // 4]
    lines.sort(key=lambda x: len(x['exists']), reverse=True)
    util.file_write_json_line(corpus_json_order_fn, lines)


def dataset_optimize():
    for k in entity_dic_jt.keys():
        entity_dic_jt[k] = 0
    if os.path.exists(final_corpus_fn):
        os.remove(final_corpus_fn)
    all_lens = 0
    with open(corpus_json_order_fn) as f:
        while True:
            lines = f.readlines(1_000_000_000)
            if len(lines) == 0:
                break
            # print(lines[0]
            data_list = []
            for line in lines:
                data_list.extend(util.line_to_json(line))
            del lines
            print("data_list ", len(data_list))
            result = []
            for jl in tqdm(data_list):
                exists = jl['exists']
                min_count = min(map(lambda x: entity_dic_jt[x], exists))
                if min_count < 10:
                    for e in exists:
                        entity_dic_jt[e] += 1
                    sentence = jl['sentence'].split('.lt;')[0]
                    sentence = sentence.replace('\n', '')
                    tokens = tokenizer.tokenize(sentence)
                    jl['tokens'] = tokens[: min(500, len(tokens))]
                    result.append(jl)
            all_lens += len(result)
            print("result", len(result))
            util.file_write_json_line(final_corpus_fn, result)
    print("all_lens", all_lens)
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
    if os.path.exists(corpus_json_fn):
        os.remove(corpus_json_fn)
    f_size = os.path.getsize(destination_fn)
    all_iteration = f_size // buffer_size
    print(f"all_iteration {all_iteration}")
    current_index = 1
    with open(destination_fn, 'r') as f:
        while True:
            f_lines = f.readlines(buffer_size)
            if len(f_lines) == 0:
                break
            # chunk_size = len(f_text) / (10000000 * number_process)
            chunk_size = 50_000
            print('start process')
            print('batch size', len(f_lines))
            # print("f_lines", f_lines[0])
            chunks = [
                f_lines[i : i + chunk_size] for i in range(0, len(f_lines), chunk_size)
            ]
            # results = [Generate_data_process(f_lines)]
            result_list = pool.map(generate_data_process_ah, chunks)
            util.file_write_json_line(corpus_json_fn, result_list)
            print(f"current/all  {current_index}/{all_iteration}")
            current_index += 1
            # util.file_write_json_line(corpus_3_fn, all_list)
        k_count = 0
        for k, v in entity_dict.items():
            if v == 0:
                k_count += 1
        print("unsen_key_count", k_count)


def test_pool():
    def sum_list(data):
        r = []
        for i in data:
            r.append(i**2)
        return r

    pool = Pool(8)
    f_lines = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    chunk_size = 2
    chunks = [f_lines[i : i + n] for i in range(0, len(f_lines), chunk_size)]
    results = pool.map(generate_data_process, chunks)


def test_tokenizer():
    line = 'Fiordland is on the west coast, but is in the Southland, New ZealandSouthland Region rather than the West Coast Region.'
    tokens = tokenizer.tokenize(line)
    print(tokens)


def sample_dev_dataset():
    val_set = util.file_read_json_line(config.VAL_FN)
    dev_set = random.sample(val_set, 1000)
    util.file_write_json_line(config.TRAIN_TINY_FN, dev_set)


# def data_analysis():


def test_ah():
    sentence = "Regions Financial Corporation is the largest bank headquartered in or operating in Alabama. PNC Financial Services and Wells Fargo also have a major presence in Alabama.lt;refgt;\n"
    s2 = "A great deal of Alabamas economic growth since the 1990s has been due to the states expanding automotive manufacturing industry. Located in the state are Honda Manufacturing of Alabama, Hyundai Motor Manufacturing Alabama, MercedesBenz U.S. International, and Toyota Motor Manufacturing Alabama, as well as their various suppliers. Since 1993, the automobile industry has generated more than 67,800 new jobs in the state. Alabama currently ranks 4th in the nation for vehicle exports.lt;refgt;\n"
    rs = kwtree.search_all(sentence)
    rs = list(rs)
    print(rs)


if __name__ == "__main__":
    # test_ah()
    # multi_process_genenrate()
    # clean_file(used_fn)
    # dataset_optimize()
    sample_dev_dataset()
    # test_tokenizer()
    # dataset_reorder()
