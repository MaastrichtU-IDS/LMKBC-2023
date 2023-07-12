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
from multiprocessing import Process
from multiprocessing import Pool

token_path = f"{config.RES_DIR}/tokenizer/bert"
tokenizer = transformers.AutoTokenizer.from_pretrained(token_path)


def build_entity_set_from_dataset(fns):
    entity_set = set()
    for fn in fns:
        with open(fn, "r") as file:
            for line in file:
                line_json = json.loads(line)
                object_entities = line_json['ObjectEntities']
                subject = line_json["SubjectEntity"]

                entity_set.add(subject)
                entity_set.update(object_entities)
    return entity_set


def extend_tokenizer(entity_set: set, tokenizer: transformers.PreTrainedTokenizerFast):
    for entity in entity_set:
        if entity not in tokenizer.get_vocab():
            tokenizer.add_tokens(entity)
    return tokenizer


def get_additional_sentence(entity_set):
    for item in entity_set:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()


def get_text(entity_list):
    content_list = []
    for entity in tqdm(entity_list):
        wiki_titles = wikipedia.search(entity)
        for title in wiki_titles:
            try:
                wiki_page = wikipedia.summary(title)
            except:
                # print("not wiki page")
                pass

            content = wiki_page
            content_list.append(content)
    local_sentence_fn = (
        f'{config.RES_DIR}/res/additional_corpus/{int(random.random*10000)}.txt'
    )
    with open(local_sentence_fn, 'w') as f:
        s = '\n'.join(content_list)
        f.writelines(s)


if __name__ == "__main__":
    # bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path="bert-base-cased"
    # )
    # entity_set = build_entity_set_from_dataset([config.TRAIN_FN, config.VAL_FN])
    # print("start building entity set")
    # tokenizer = extend_tokenizer(entity_set, bert_tokenizer)
    # print("start extend tokenizer")
    # tokenizer_path = f'{config.RES_PATH}/tokenizer/bert'
    # tokenizer.save_pretrained(tokenizer_path)
    # print("save tokenizer success")
    local_sentence_fn = f'{config.RES_DIR}/sentence.txt'
    sentence_list = []
    with open(f'{config.RES_DIR}/tokenizer/bert/added_tokens.json') as f:
        entity_dict = json.load(f)
    entity_set = entity_dict.keys()
    entity_set = list(entity_set)
    worker_size = 10
    with Pool(processes=worker_size) as pool:
        pool.map(get_text, entity_set, chunksize=len(entity_set) / worker_size)
