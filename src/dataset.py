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


class MLMDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizerFast, data_fn, template_fn) -> None:
        super().__init__()

        base_name = os.path.basename(data_fn)
        # base_dir = os.path.dirname(data_fn)
        pickle_fn = f'{config.DATA_DIR}\\store_{base_name}.pickle'
        if os.path.exists(pickle_fn):
            with open(pickle_fn, 'rb') as f:
                self.data = pickle.load(f)
        else:
            max_sentence_length = 0
            max_obj_length = 0
            self.data = []
            with open(data_fn, "r") as file:
                train_data = [json.loads(line) for line in file]
            with open(template_fn, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                prompt_templates = {
                    row['Relation']: row['PromptTemplate'] for row in reader
                }
            mask_pad = [tokenizer.mask_token] * config.MASK_TOKEN_SIZE
            for row in train_data:
                relation = row['Relation']
                prompt_template = prompt_templates[relation]
                object_entities = row['ObjectEntities']
                subject = row["SubjectEntity"]
                # if subject not in tokenizer.vocab:
                #     tokenizer.add_tokens(subject)
                for obj in object_entities:
                    # if obj not in tokenizer.vocab:
                    #     tokenizer.add_tokens(obj)
                    word_tokens = tokenizer.tokenize(obj)
                    if len(word_tokens) > max_obj_length:
                        max_obj_length = len(word_tokens)
                    word_tokens.extend(
                        [tokenizer.pad_token]
                        * (config.MASK_TOKEN_SIZE - len(word_tokens))
                    )
                    # word_pad, mask_pad = mask_word_func(obj, tokenizer)
                    # label_sentence = prompt_template.format(
                    #     subject_entity=subject, mask_token=word_pad
                    # )
                    # obj_token = tokenizer.tokenize(obj)
                    # mask_token = " ".join([tokenizer.mask_token] * len(obj_token))

                    input_sentence = prompt_template.format(
                        subject_entity=subject, mask_token=tokenizer.mask_token
                    )
                    # label_tokens = tokenizer.tokenize(label_sentence)

                    input_tokens = tokenizer.tokenize(input_sentence)

                    mask_input_tokens = list_replace_func(
                        input_tokens, tokenizer.mask_token, mask_pad
                    )
                    mask_label_tokens = list_replace_func(
                        input_tokens, tokenizer.mask_token, word_tokens
                    )
                    label_ids = tokenizer.convert_tokens_to_ids(mask_label_tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(mask_input_tokens)
                    attention_mask = [
                        0 if v == tokenizer.mask_token else 1 for v in mask_input_tokens
                    ]
                    if len(label_ids) != len(input_ids):
                        print("mask_label_tokens", mask_label_tokens)
                        print("mask_input_tokens", mask_input_tokens)
                        print("word_tokens length", len(word_tokens))
                        raise Exception("length of label and input is not equal")

                    if len(input_ids) > max_sentence_length:
                        max_sentence_length = len(input_ids)

                    item = {
                        "labels": label_ids,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
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


def mask_word_func(obj, tokenizer: BertTokenizer):
    word_tokens = tokenizer.tokenize(obj)
    mask_pad = [tokenizer.mask_token] * config.MASK_TOKEN_SIZE
    word_tokens.extend(
        [tokenizer.pad_token] * (config.MASK_TOKEN_SIZE - len(word_tokens))
    )
    return word_tokens, mask_pad


def list_replace_func(origin_list: list, old_item, new_item):
    if not isinstance(origin_list, list):
        raise Exception("origin_list must be a list")

    if not isinstance(old_item, list):
        old_item = [old_item]
    if not isinstance(new_item, list):
        new_item = [new_item]
    old_index = None
    for i in range(len(origin_list)):
        if (
            origin_list[i] == old_item[0]
            and origin_list[i : i + len(old_item)] == old_item
        ):
            old_index = i
            break
    if old_index is None:
        raise Exception("old item is not found in origin_list")
    result_list = (
        origin_list[:old_index] + new_item + origin_list[old_index + len(old_item) :]
    )
    # print(result_list)
    return result_list


def template(data_fn, template_fn, tokenizer: BertTokenizer):
    with open(data_fn, "r") as file:
        train_data = [json.loads(line) for line in file]
    with open(template_fn, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}

    for row in train_data:
        relation = row['Relation']
        prompt_template = prompt_templates[relation]
        object_entities = row['ObjectEntities']
        subject = row["SubjectEntity"]
        # if subject not in tokenizer.vocab:
        #     tokenizer.add_tokens(subject)
        for obj in object_entities:
            if obj not in tokenizer.vocab:
                tokenizer.add_tokens(obj)

            label_sentence = prompt_template.format(
                subject_entity=subject, mask_token=obj
            )
            # obj_token = tokenizer.tokenize(obj)
            # mask_token = " ".join([tokenizer.mask_token] * len(obj_token))

            input_sentence = prompt_template.format(
                subject_entity=subject, mask_token=tokenizer.mask_token
            )
            label_tokens = tokenizer.tokenize(label_sentence)
            input_tokens = tokenizer.tokenize(input_sentence)

            label_ids = tokenizer.convert_tokens_to_ids(label_tokens)
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

            assert len(label_ids) == len(input_ids)
            item = {
                "labels": label_ids,
                "input_ids": input_ids,
            }


if __name__ == "__main__":
    bert_tokenizer: BertTokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="bert-base-cased"
    )
    word = "sdfsdgdd"
    word_tokens = bert_tokenizer.tokenize(word)
    print(word_tokens)
    # word_pad = bert_tokenizer.pad(word_tokens, max_length=10)
    word_pad = word_tokens.extend([bert_tokenizer.pad_token] * (10 - len(word_tokens)))
    print('word_pad', word_pad)
    word_resume = bert_tokenizer.convert_tokens_to_string(word_tokens)
    print(word_resume)

    index_padding = word.find("1")
    print(index_padding)

    index_list = word_tokens.extend('s')
    print(index_list)

    for i in range(3):
        word_tokens.insert(i, i)
    print(word_tokens)
