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


class MLMDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizerFast, data_fn, template_fn) -> None:
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
            train_data = util.file_read_train(data_fn)
            prompt_templates = util.file_read_prompt(template_fn)

            for row in train_data:
                relation = row['Relation']
                prompt_template = prompt_templates[relation]
                object_entities = row['ObjectEntities']
                subject = row["SubjectEntity"]
                if subject not in tokenizer.vocab:
                    # print("add token ", obj)
                    tokenizer.add_tokens(subject)
                for obj in object_entities:
                    if obj == '':
                        obj = config.EMPTY_TOKEN
                    if obj not in tokenizer.vocab:
                        tokenizer.add_tokens(obj)
                    label_sentence = prompt_template.format(
                        subject_entity=subject, mask_token=obj
                    )
                    input_sentence = prompt_template.format(
                        subject_entity=subject, mask_token=tokenizer.mask_token
                    )
                    label_tokens = tokenizer.tokenize(label_sentence)

                    input_tokens = tokenizer.tokenize(input_sentence)
                    label_ids = tokenizer.convert_tokens_to_ids(label_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                    attention_mask = [
                        0 if v == tokenizer.mask_token else 1 for v in input_tokens
                    ]
                    # encoded_sentence = tokenizer.encode_plus(text=input_sentence)
                    if len(label_ids) != len(input_ids):
                        print("label_sentence ", label_sentence)
                        print("input_sentence ", input_sentence)

                        print("label_tokens ", label_tokens)
                        print("input_tokens ", input_tokens)
                        print("obj ", obj)
                        # print(obj in tokenizer.get_vocab())
                        # tokenizer.save_vocabulary(config.OUTPUT_DIR, "vocab.txt")

                        # enco_o = tokenizer.encode_plus(text=label_sentence)
                        # print("enco_o ", enco_o)
                        print(tokenizer.add_tokens(obj))
                        print(tokenizer.tokenize(obj))
                        # print("word_tokens length", len(word_tokens))
                        continue
                        # raise Exception("length of label and input is not equal")

                    if len(input_ids) > max_sentence_length:
                        max_sentence_length = len(input_ids)

                    item = {
                        "labels": label_ids,
                        "input_ids": input_ids,
                        # "attention_mask": attention_mask,
                    }
                    self.data.append(item)
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
