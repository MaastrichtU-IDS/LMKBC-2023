import csv
import json
import os
import random
from typing import List

import requests

import config

local_cache_path = f'{config.DATA_DIR}\\item_cache.json'
local_cache = dict()
if os.path.exists(local_cache_path):
    local_cache = json.load(open(local_cache_path))


# Disambiguation baseline
def disambiguation_baseline(item):
    if item in local_cache:
        return local_cache[item]
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()
        # Return the first id (Could upgrade this in the future)
        first_id = data['search'][0]['id']
        local_cache[item] = first_id
        with open(local_cache_path, "w") as f:
            json.dump(local_cache, f)
        return first_id
    except:
        return item


# Read prompt templates from a CSV file
def file_read_prompt(file_path: str):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
    return prompt_templates


def file_read_train(data_fn):
    with open(data_fn, "r") as file:
        train_data = [json.loads(line) for line in file]
        return train_data


def file_write_json_line(data_fn, results):
    with open(data_fn, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


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
