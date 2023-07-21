import json
import os
import random
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)


import config
import util
import transformers


with open(f'{config.RES_DIR}/tokenizer/bert/added_tokens.json') as f:
    entity_dic_jt = json.load(f)

train_line = util.file_read_json_line(config.TRAIN_FN)
val_line = util.file_read_json_line(config.VAL_FN)
all_line = train_line + val_line


def count_object():
    null_count = 0
    for line in all_line:
        obj_list = line[config.KEY_OBJS]
        if len(obj_list) == 0:
            null_count += 1
    print("null_count", null_count)


def count_emapt():
    empty_count = 0
    for line in all_line:
        obj_list = line[config.KEY_OBJS]
        for obj in obj_list:
            if obj == config.EMPTY_TOKEN:
                empty_count += 1
    print("empty_count", empty_count)


def count_emapt_and_other():
    empty_count = 0
    for line in all_line:
        obj_list = line[config.KEY_OBJS]
        if '' in obj_list and len(obj_list) > 1:
            empty_count += 1
    print("empty_count", empty_count)


def get_all_entity(fn = None):
    if fn is not None:
        lines = util.file_read_json_line(fn)
    else:
        lines = all_line
    entity_set = set()
    for row in lines:
        object_entities = row['ObjectEntities']
        subject = row["SubjectEntity"]
        entity_set.add(subject)
        entity_set.update(object_entities)
    return entity_set


def collect_entity():
    entity_fn = f'{config.RES_DIR}/additional_corpus/entity.txt'
    # entity_1_set = [t.replace("\"", "") for t in list(entity_set)]
    util.file_write_line(entity_fn, list(entity_dic_jt.keys()), mode='w')

def negave_vocabulary():
    true_entity = get_all_entity()
    
    output_fm_train = "output/filled-mask/filled-mask-train.jsonl"
    output_fm_valid = "output/filled-mask/filled-mask-valid.jsonl"
    predict_entity_train = get_all_entity(output_fm_train)
    predict_entity_valid = get_all_entity(output_fm_valid)
    false_entity =(predict_entity_train| predict_entity_valid) -  true_entity
    print(false_entity)
    entity_fn = f'data/negave_vocabulary.json'
    with open(entity_fn,'w') as f:
        json.dump(list(false_entity),f)



if __name__ == "__main__":
    # count_object()
    # count_emapt_and_other()
    # collect_entity()
    negave_vocabulary()
