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

pickle_dir = f'{config.RES_PATH}\\pickle'
if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)

prompt = '{subject} is {relation} with {object}'


class NSPDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizerFast, data_fn) -> None:
        super().__init__()

        base_name = os.path.basename(data_fn)
        # base_dir = os.path.dirname(data_fn)
        pickle_fn = f'{pickle_dir}\\nsp_{base_name}.pickle'
        self.tokenizer = tokenizer
        if os.path.exists(pickle_fn) and False:
            with open(pickle_fn, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = []
            # print(data_fn)
            train_data = util.file_read_train(data_fn)
            # print(train_data[0])
            max_length = 0
            for row in train_data:
                given_subject = row['given_subject']
                given_object = row['given_object']
                triple = row['triple']
                label = row['label']
                subject_sentence = self.given_to_sentence(given_subject)
                object_sentence = self.given_to_sentence(given_object)
                triple_sentence = self.triple_to_sentenct(*triple)
                text = f"{subject_sentence} while {object_sentence}"
                encoded = tokenizer.encode_plus(
                    text=text,
                    text_pair=triple_sentence,
                    return_tensors='pt',
                    padding="max_length",
                    max_length=config.MAX_LENGTH,
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
            with open(pickle_fn, 'wb') as f:
                pickle.dump(self.data, f)
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
                sentence = self.triple_to_sentenct(subject, relation, obj)
                sentence_list.append(sentence)
        result = ";".join(sentence_list)
        return result

    def triple_to_sentenct(self, subject, relation, object):
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