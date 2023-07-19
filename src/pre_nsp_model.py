import csv
import json
import argparse
import logging
import random
import copy
import numpy as np
import pandas as pd
import requests

import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    pipeline,
    BertTokenizer,
    BertModel,
)
import os
import config
from evaluate import combine_scores_per_relation, evaluate_per_sr_pair
from file_io import read_lm_kbc_jsonl
import util

from sklearn import metrics

task = 'nsp'
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


prompt = '{subject} has {relation} with {object}'


'''
{
    "SubjectEntityID": "Q225",
    "SubjectEntity": "Bosnia and Herzegovina",
    "ObjectEntitiesID": [        "Q115",        "Q1036"    ],
    "ObjectEntities": [        "Ethiopia",        "Uganda"    ],
    "Relation": "CountryBordersCountry",
    "ObjectLabels": [        0,        1    ],
    "TrueObjectEntities": [        "Uganda"    ]
}
'''


class NSPDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, data_fn, kg) -> None:
        super().__init__()
        self.data = []
        self.kg = kg
        train_data = util.file_read_json_line(data_fn)
        self.miss_entity_count = 0
        for row in train_data:
            object_entities = row[config.KEY_OBJS]
            object_labels = row[config.OBJLABELS_KEY]
            relation = row[config.KEY_REL]
            subject = row[config.KEY_SUB]
            subject_context = self.generate_sentence_from_context(subject)
            for obj, label in zip(object_entities, object_labels):
                triple_sentence = self.triple_to_sentence(subject, relation, obj)
                encoded = tokenizer.encode_plus(
                    text=subject_context,
                    text_pair=triple_sentence,
                )
                encoded['labels'] = label
                self.data.append(encoded)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def generate_sentence_from_context(self, entity: str):
        # entity->from/to->relation->objects
        if entity not in self.kg:
            self.miss_entity_count += 1
            return 'no context'
        sentence_list = []
        from_relations = self.kg[entity][config.FROM_KG]
        for relation, objects in from_relations.items():
            for obj in objects:
                if len(sentence_list) > 25:
                    continue
                sentence = prompt.format(subject=obj, relation=relation, object=entity)
                sentence_list.append(sentence)

        to_relations = self.kg[entity][config.TO_KG]
        for relation, objects in to_relations.items():
            for obj in objects:
                if len(sentence_list) > 25:
                    continue
                sentence = prompt.format(subject=entity, relation=relation, object=obj)
                sentence_list.append(sentence)
        return ' and '.join(sentence_list)

    def triple_to_sentence(self, subject, relation, obj):
        sentence = prompt.format(subject=subject, relation=relation, object=obj)
        return sentence


def train():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_load_dir)
    bert_model: BertModel = transformers.BertForNextSentencePrediction.from_pretrained(
        args.model_load_dir, config=bert_config
    )

    # bert_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_load_dir)

    train_dataset = NSPDataset(tokenizer=bert_tokenizer, data_fn=args.train_fn, kg=kg)
    print(f"train dataset size: {len(train_dataset)}")
    dev_dataset = NSPDataset(tokenizer=bert_tokenizer, data_fn=args.dev_fn, kg=kg)

    bert_model.resize_token_embeddings(len(bert_tokenizer))

    training_args = transformers.TrainingArguments(
        output_dir=args.model_save_dir,
        overwrite_output_dir=True,
        # evaluation_strategy='epoch',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=64,
        eval_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.train_epoch,
        warmup_ratio=0.1,
        logging_dir=config.LOGGING_DIR,
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        fp16=True,
        # dataloader_num_workers=0,
        # auto_find_batch_size=False,
        # greater_is_better=False,
        # load_best_model_at_end=True,
    )

    trainer = transformers.Trainer(
        model=bert_model,
        # data_collator=bert_collator,
        train_dataset=train_dataset,
        args=training_args,
        # eval_dataset=dev_dataset,
        tokenizer=bert_tokenizer,
    )
    # compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir=args.model_best_dir)
    # bert_tokenizer.save_pretrained(args.bin_dir)
    # dev_results = trainer.evaluate(dev_dataset)
    # # trainer.model
    # print(f"dev results: ")
    # print(dev_results)


def evaluate():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)
    bert_model: BertModel = transformers.BertForNextSentencePrediction.from_pretrained(
        args.model_best_dir, config=bert_config
    )

    dev_dataset = NSPDataset(data_fn=args.test_fn, tokenizer=bert_tokenizer, kg=kg)
    bert_model.resize_token_embeddings(len(bert_tokenizer))

    training_args = transformers.TrainingArguments(
        output_dir=args.model_save_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=64,
        eval_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.train_epoch,
        warmup_ratio=0.1,
        logging_dir=config.LOGGING_DIR,
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=0,
        auto_find_batch_size=False,
        greater_is_better=False,
        load_best_model_at_end=True,
        no_cuda=False,
    )

    trainer = transformers.Trainer(
        model=bert_model,
        args=training_args,
        # eval_dataset=dev_dataset,
        tokenizer=bert_tokenizer,
    )

    predicts = trainer.predict(dev_dataset)
    # trainer.evaluate()

    predictions = util.softmax(predicts.predictions, axis=1)
    pre_score = predictions[:, 1]
    print(pre_score[:10])
    pre = np.where(pre_score > 0.99, 1, 0)
    label_ids = predicts.label_ids.reshape(-1, 1)
    print("pre shape", pre.shape)
    print("label_ids shape", label_ids.shape)
    print("positive label", np.sum(label_ids == 1))
    print("positive pre", np.sum(pre == 1))
    precision = metrics.precision_score(label_ids, pre)
    recall = metrics.recall_score(label_ids, pre)
    f1 = metrics.f1_score(label_ids, pre)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)


def test():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)
    bert_model: BertModel = transformers.BertForNextSentencePrediction.from_pretrained(
        args.model_best_dir, config=bert_config
    )

    test_dataset = NSPDataset(data_fn=args.test_fn, tokenizer=bert_tokenizer)
    bert_model.resize_token_embeddings(len(bert_tokenizer))

    training_args = transformers.TrainingArguments(
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=64,
        eval_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.train_epoch,
        warmup_ratio=0.1,
        logging_dir=config.LOGGING_DIR,
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=0,
        auto_find_batch_size=False,
        greater_is_better=False,
        load_best_model_at_end=True,
        no_cuda=False,
    )

    trainer = transformers.Trainer(
        model=bert_model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=bert_tokenizer,
    )

    predicts = trainer.predict(test_dataset)
    print(predicts)
    pre = np.argmax(predicts.predictions, axis=-1)
    label_ids = predicts.label_ids.reshape(-1, 1)
    print("pre shape", pre.shape)
    print("label_ids shape", label_ids.shape)
    print("positive label", np.sum(label_ids == 1))
    print("positive pre", np.sum(pre == 1))
    precision = metrics.precision_score(label_ids, pre)
    recall = metrics.recall_score(label_ids, pre)
    f1 = metrics.f1_score(label_ids, pre)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Model with Question and Fill-Mask Prompts"
    )

    parser.add_argument(
        "--model_save_dir",
        type=str,
        help="HuggingFace model name (default: bert-base-cased)",
    )
    parser.add_argument(
        "--model_best_dir",
        type=str,
        help="HuggingFace model name (default: bert-base-cased)",
    )

    parser.add_argument(
        "--model_load_dir",
        type=str,
        help="HuggingFace model name (default: bert-base-cased)",
    )

    # parser.add_argument(
    #     "--pretrin_model",
    #     type=str,
    #     help="HuggingFace model name (default: bert-base-cased)",
    # )

    parser.add_argument(
        "-i", "--test_fn", type=str, required=True, help="Input test file (required)"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output file (required)"
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=100,
        help="Top k prompt outputs (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        help="Probability threshold (default: 0.5)",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=4e-5,
        help="Probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help="GPU ID, (default: -1, i.e., using CPU)",
    )

    parser.add_argument(
        "-fp",
        "--template_fn",
        type=str,
        required=True,
        help="CSV file containing fill-mask prompt templates (required)",
    )

    parser.add_argument(
        "--train_fn",
        type=str,
        required=True,
        help="CSV file containing train data for few-shot examples (required)",
    )
    parser.add_argument(
        "--train_epoch",
        type=int,
        default=10,
        help="CSV file containing train data for few-shot examples (required)",
    )

    parser.add_argument(
        "--dev_fn",
        type=str,
        required=True,
        help="CSV file containing train data for few-shot examples (required)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for the model. (default:32)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train eval test",
        help="Batch size for the model. (default:32)",
    )

    args = parser.parse_args()
    # tokenizer_dir = f'{config.RES_DIR}/tokenizer/bert'
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=bert_tokenizer
    )
    kg = util.KnowledgeGraph(args.train_fn)
    if "train" in args.mode:
        train()

    if "test" in args.mode:
        evaluate()
