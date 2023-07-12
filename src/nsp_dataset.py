import csv
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
    BertTokenizer,
    BertTokenizerFast,
)

import transformers
import pickle
import config
import util

pickle_dir = f'{config.RES_DIR}\\pickle'
if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)

prompt = '{subject} is {relation} with {object}'


class NSPDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizerFast, data_fn, kg_fn) -> None:
        super().__init__()

        self.data = []
        # print(data_fn)
        train_data = util.file_read_json_line(data_fn)
        # print(train_data[0])
        max_length = 0
        kg = util.build_knowledge_graph(data_fn=kg_fn)
        for row in train_data:
            triple = row['triple']
            label = row['label']
            subject_sentence = self.given_to_sentence(given_subject)
            object_sentence = self.given_to_sentence(given_object)
            triple_sentence = self.triple_to_sentence(*triple)
            text = f"{subject_sentence} while {object_sentence}"
            encoded = tokenizer.encode_plus(
                text=text,
                text_pair=triple_sentence,
                return_tensors='pt',
                padding="max_length",
                max_length=config.FM_MAX_LENGTH,
            )
            input_ids = encoded['input_ids'].squeeze()
            attention_mask = encoded['attention_mask'].squeeze()
            token_type_ids = encoded['token_type_ids'].squeeze()
            if (l := len(input_ids)) > max_length:
                max_length = l
            item = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                "label": torch.tensor(label),
            }

            self.data.append(item)
        print("max_sentence_length", max_length)
        # print("max_obj_length", max_obj_length)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def given_to_sentence(self, given_condition):
        if len(given_condition) < 3:
            return ""
        subject = given_condition[0]

        sentence_list = []
        for d in given_condition[1]:
            relation = d[config.KEY_REL]
            objs = d[config.KEY_OBJS]
            for obj in objs:
                sentence = self.triple_to_sentence(subject, relation, obj)
                sentence_list.append(sentence)
        result = ";".join(sentence_list)
        return result

    def triple_to_sentence(self, subject, relation, object):
        if len(subject) > 20:
            self.tokenizer.add_tokens(subject)
        if len(object) > 20:
            self.tokenizer.add_tokens(object)
        sentence = prompt.format(subject=subject, relation=relation, object=object)
        return sentence
        # "given_subject": [
        #     "Hafei Motor Co.",
        #     [
        #         {
        #             "Relation": "CompanyHasParentOrganisation",
        #             "ObjectEntities": [
        #                 "Changan Automobile"
        #             ]
        #         }
        #     ]
        # ],
