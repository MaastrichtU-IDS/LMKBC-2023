import csv
import json
import os
import random
from typing import List

import requests
import torch
from transformers import BertTokenizerFast

import config as config

from typing import List, Union, Dict, Any, Optional, Mapping
import numpy as np

local_cache_path = f'{config.RES_DIR}/item_cache.json'
local_cache = dict()
if os.path.exists(local_cache_path):
    local_cache = json.load(open(local_cache_path))


# Disambiguation baseline
def disambiguation_baseline(entity_label):
    if entity_label in local_cache:
        return local_cache[entity_label]
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entity_label}&language=en&format=json"
        data = requests.get(url).json()
        # Return the first id (Could upgrade this in the future)
        first_id = data['search'][0]['id']
        local_cache[entity_label] = first_id
        return first_id
    except:
        return entity_label
    
def save_entity_id():
    with open(local_cache_path, "w") as f:
        json.dump(local_cache, f, indent = 2)



# Read prompt templates from a CSV file
def file_read_prompt(file_path: str):
    # print('file_path', file_path)
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
    return prompt_templates


def line_to_json(line: str):
    train_data = []
    if len(line) == 0:
        return train_data
    try:
        if line.find('}{') != -1:
            line = line.replace('}{', '}\n{')
            line_list = line.split('\n')
            for l in line_list:
                train_data.extend(line_to_json(l))
        else:
            train_data.append(json.loads(line))
    except Exception as e:
        print(line)
        raise e
    return train_data


def file_read_json_line(data_fn):
    train_data = []
    try:
        with open(data_fn, "r") as file:
            lines = file.readlines()
            for line in lines:
                train_data.append(json.loads (line))
    except :
        print(line)

    return train_data


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def flat_list(data_list: list):
    datas = []
    for data in data_list:
        if isinstance(data, list):
            datas.extend(flat_list(data))
        else:
            datas.append(data)
    return datas


def file_write_json_line(data_fn, results, mode='w'):
    results = flat_list(results)
    json_text_list = [json.dumps(aj, cls=SetEncoder) for aj in results]
    file_write_line(data_fn, json_text_list, mode)

def file_delete(fp):
    if os.path.exists(fp):
        os.remove(fp)

def file_write_line(data_fn, results, mode='w'):
    if mode == 'auto':
        mode = 'a' if os.path.exists(data_fn) else 'w'
    with open(data_fn, mode) as f:
        text = '\n'.join(results)
        if mode == 'a':
            text = '\n'+text
        f.write(text)


def create_prompt(
    subject_entity: str,
    relation: str,
    prompt_templates: dict,
    instantiated_templates: List[str],
    tokenizer,
    few_shot: int = 0,
    task: str = "fill-mask",
) -> str:
    prompt_template = prompt_templates[relation]
    if task == "text-generation":
        if few_shot > 0:
            random_examples = random.sample(
                instantiated_templates, min(few_shot, len(instantiated_templates))
            )
        else:
            random_examples = []
        few_shot_examples = "\n".join(random_examples)
        prompt = f"{few_shot_examples}\n{prompt_template.format(subject_entity=subject_entity)}"
    else:
        prompt = prompt_template.format(
            subject_entity=subject_entity, mask_token=tokenizer.mask_token
        )
    return prompt


def recover_mask_word_func(mask_word, bert_tokenizer):
    word_resume = bert_tokenizer.convert_tokens_to_string(mask_word)
    index_padding = word_resume.find(bert_tokenizer.padding_token)
    if index_padding > -1:
        word_resume = word_resume[:index_padding]
    return word_resume


class KnowledgeGraph:
    def __init__(self, data_fn, kg=None):
        # read train file, each line is a josn object
        train_line = file_read_json_line(data_fn)
        # if parameter kg is none, create a new dict object. 
        # else use the parameter kg. 
        self.kg = dict() if kg is None else kg
        for row in train_line:
            relation = row['Relation']
            object_entities = row['ObjectEntities']
            subject = row["SubjectEntity"]
            self.add_triple(subject, relation, object_entities)

    def ensure_key_exists_for_entity(self, entity, relation):
        # make sure the basic key for each entity exists
        if entity not in self.kg:
            self.kg[entity] = dict()
        if config.TO_KG not in self.kg[entity]:
            self.kg[entity][config.TO_KG] = dict()
        if config.FROM_KG not in self.kg[entity]:
            self.kg[entity][config.FROM_KG] = dict()

        if relation not in self.kg[entity][config.TO_KG]:
            self.kg[entity][config.TO_KG][relation] = set()
        if relation not in self.kg[entity][config.FROM_KG]:
            self.kg[entity][config.FROM_KG][relation] = set()

    def add_triple(self, subject, relation, object_entities):
        # for example: 
        # {
        #     "The Netherlands": {
        #         "to": {
        #             "CountryBordersCountry": [
        #                 "Germany"
        #             ]
        #         },
        #         "from": {
        #             "CountryBordersCountry": [
        #                 "Belgium"
        #             ]
        #         }
        #     }
        # }
        if subject == config.EMPTY_STR:
            return 
        if not isinstance(object_entities, (list, set)):
            object_entities = [object_entities]

        self.ensure_key_exists_for_entity(subject, relation)
        self.kg[subject][config.TO_KG][relation].update(object_entities)
        for entity in object_entities:
            if entity == '':
                continue 
            self.ensure_key_exists_for_entity(entity, relation)
            self.kg[entity][config.FROM_KG][relation].add(subject)

    def __getitem__(self, index):
        if index in self.kg:
            return self.kg[index]
        else:
            return None

    def __contains__(self, item):
        return item in self.kg


class DataCollatorKBC:
    def __init__(self, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # examples = examples.clone()
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_attention_mask=True,
        )
        max_length = len(batch['input_ids'][0])
        label_list = []
        for label in batch['labels']:
            length_diff = max_length - len(label)
            label1 = label.copy()
            if length_diff > 0:
                pad_list = [-100] * (length_diff)
                label1.extend(pad_list)

            label_list.append(label1)
        batch['labels'] = label_list
        batch_pt = dict()
        for k, v in batch.items():
            batch_pt[k] = torch.tensor(v)

        return batch_pt


def tokenize_sentence(tokenizer, input_sentence: str):
    input_tokens = (
        [tokenizer.cls_token]
        + tokenizer.tokenize(input_sentence)
        + [tokenizer.sep_token]
    )
    # input_tokens = tokenizer.tokenize(input_sentence)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [0 if v == tokenizer.mask_token else 1 for v in input_tokens]

    return input_ids, attention_mask


def softmax(x, axis=0):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


if __name__ == "__main__":
    list_1 = [[1], [1, 2], [[1], [2], [2, 3, [4, 5, [7]]]]]
    list_flat = flat_list(list_1)
    print(list_flat)


class Printer:
    def __init__(self, times):
        self.times=times
        self.channel_times=dict()

    def __call__(self, obj, channel='default'):
        if channel not in self.channel_times:
            self.channel_times[channel]=self.times
        if self.channel_times[channel] > 0:
            print(obj)
            self.channel_times[channel]-=1
