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


class GEDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_fn, template_fn) -> None:
        super().__init__()

        base_name = os.path.basename(data_fn)
        # base_dir = os.path.dirname(data_fn)
        pickle_fn = f'{config.RES_PATH}/fm_{base_name}.pickle'
        if os.path.exists(pickle_fn) and False:
            with open(pickle_fn, 'rb') as f:
                self.data = pickle.load(f)
        else:
            max_sentence_length = 0
            max_obj_length = 0
            self.data = []
            train_data = util.file_read_json_line(data_fn)
            prompt_templates = util.file_read_prompt(template_fn)

            for row in train_data:
                relation = row['Relation']
                prompt_template = prompt_templates[relation]
                object_entities = row['ObjectEntities']
                subject = row["SubjectEntity"]
                # if subject not in tokenizer.vocab:
                #     # print("add token ", obj)
                #     tokenizer.add_tokens(subject)
                relation = row['Relation']
                prompt_template = prompt_templates[relation]
                object_entities = row['ObjectEntities']
                # for obj in object_entities:
                #     if obj not in tokenizer.vocab:
                #         tokenizer.add_tokens(obj)
                answers = ','.join(object_entities)
                instantiated_example = (
                    prompt_template.format(subject_entity=subject) + f" {answers}"
                )
                encoded = tokenizer.encode_plus(instantiated_example)
                # print("encoded", encoded)
                if len(encoded['input_ids']) > max_sentence_length:
                    max_sentence_length = len(encoded['input_ids'])
                self.data.append(encoded)
            with open(pickle_fn, 'wb') as f:
                pickle.dump(self.data, f)
            print("max_sentence_length", max_sentence_length)
            print("max_obj_length", max_obj_length)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    bert_tokenizer: BertTokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="bert-base-cased"
    )
    # word = "sdfsdgdd"
    # word_tokens = bert_tokenizer.tokenize(word)
    # print(word_tokens)
    # # word_pad = bert_tokenizer.pad(word_tokens, max_length=10)
    # word_pad = word_tokens.extend([bert_tokenizer.pad_token] * (10 - len(word_tokens)))
    # print('word_pad', word_pad)
    # word_resume = bert_tokenizer.convert_tokens_to_string(word_tokens)
    # print(word_resume)

    # index_padding = word.find("1")
    # print(index_padding)

    # index_list = word_tokens.extend('s')
    # print(index_list)

    # for i in range(3):
    #     word_tokens.insert(i, i)
    # print(word_tokens)
    text = "Kara Darya river basin is located in Uzbekistan."
    if "Uzbekistan" not in bert_tokenizer.vocab:
        bert_tokenizer.add_tokens("Uzbekistan")

    encode_plus_o = bert_tokenizer.encode_plus(text=text)

    string = bert_tokenizer.convert_ids_to_tokens(encode_plus_o.input_ids)
    print(encode_plus_o)
    print(string)
