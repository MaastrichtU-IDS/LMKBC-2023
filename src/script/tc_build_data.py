import os
import random
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)

import config
import util


def build_knowledge_graph(data_fn):
    train_line = util.file_read_train(data_fn)
    print('length of file ', len(train_line))
    kg = dict()
    kg_count = dict()
    relation_set = set()
    entity_set = set()
    for row in train_line:
        relation = row['Relation']
        object_entities = row['ObjectEntities']
        subject = row["SubjectEntity"]
        relation_set.add(relation)
        entity_set.add(subject)
        entity_set.update(object_entities)
        key_subject_relation = (subject, relation)
        if key_subject_relation not in kg:
            kg[subject] = list()
            kg_count[subject] = 0
        value = (relation, tuple(object_entities))
        item = {config.KEY_REL: relation, config.KEY_OBJS: object_entities}
        kg[subject].append(item)
        kg_count[subject] += 1
    print('relation set', relation_set)
    print('length of subjects ', len(kg))
    print('length of entity set', len(entity_set))
    return relation_set, kg


def build_adversarial_corpus(kg: dict, relation_set):
    data_list = []
    for subject, value in kg.items():
        given_subject = (subject, value)
        relation_of_subject_set = {v[config.KEY_REL] for v in value}
        for item in value:
            relation = item[config.KEY_REL]
            obj_set = item[config.KEY_OBJS]
            for obj in obj_set:
                if obj not in kg:
                    if random.random() < 0.6:
                        continue
                    given_object = ""
                else:
                    given_object = (obj, kg[obj])
                triple = (subject, relation, obj)
                item = {
                    "given_subject": given_subject,
                    "given_object": given_object,
                    "triple": triple,
                    "label": 1,
                }
                data_list.append(item)

                for r in relation_set:
                    if r not in relation_of_subject_set:
                        if random.random() < 0.9:
                            continue
                        fake_triple = (subject, r, obj)
                        item = {
                            "given_subject": given_subject,
                            "given_object": given_object,
                            "triple": fake_triple,
                            "label": 0,
                        }
                        data_list.append(item)

                # 1 mears true, while 0 means false
            # print(subject, relation, obj)
    print("length of data list is ", len(data_list))
    # print(data_list[:3])
    random.shuffle(data_list)
    return data_list

def generate_corpus(data_path, new_fn):
    relation_set, kg = build_knowledge_graph(data_path)
    corpus = build_adversarial_corpus(kg, relation_set)
    positive_number = 0
    for item in corpus:
        if item['label'] == 1:
            positive_number += 1
    print("positive_number", positive_number)
    print()
    data_fn = f"{config.DATA_DIR}/{new_fn}"
    util.file_write_json_line(data_fn, corpus)

if __name__ == "__main__":

    generate_corpus(config.VAL_FN, "triple_classification_val.jsonl" )
    generate_corpus(config.TRAIN_FN, "triple_classification_train.jsonl" )
    generate_corpus(config.TRAIN_TINY_FN, "triple_classification_dev.jsonl" )

