import json
import os
import sys
import requests
import transformers
import wikipedia

parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)

import config
import util


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

    wiki = wikipedia.summary("province of China", sentences=200)
    print(wiki)
