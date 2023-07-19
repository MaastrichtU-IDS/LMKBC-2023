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
example of lines of the file of train set
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
    def __init__(self, tokenizer: BertTokenizer, data_fn, kg, template_fn) -> None:
        super().__init__()
        self.data = []  
        self.kg = kg 
        train_data = util.file_read_json_line(
            data_fn
        )  # Read the training data from the file
        self.miss_entity_count = 0  # Initialize a counter for missing entities
        self.prompt_templates = util.file_read_prompt(
            template_fn
        )  # Read prompt templates from a file

        # Iterate over each row in the training data
        for row in train_data:
            object_entities = row[config.KEY_OBJS]
            object_labels = row[config.OBJLABELS_KEY]
            relation = row[config.KEY_REL]
            subject = row[config.KEY_SUB]
            # generate the contenx sentence of a giving subject entity
            subject_context = self.generate_sentence_from_context(subject)
            prompt = self.prompt_templates[relation]
            # Iterate over each object entity and label in parallel
            for obj, label in zip(object_entities, object_labels):
                # translate a triple into a sentence using a prompt
                # a example of prompt:
                # {subject_entity} is a city located at the {mask_token} river.
                # the Netherlands borders Germany, Belgium and Demark 
                query_sentence = prompt.format(
                    subject_entity=subject,mask_token=obj
                )
                # generate the contenx sentence of a giving object entity
                object_context = self.generate_sentence_from_context(obj)
                context = f"{subject_context} while {object_context}"
                encoded = tokenizer.encode_plus(
                    text=context,
                    text_pair=query_sentence,
                )

                # Truncate encoded inputs if the length exceeds 500 tokens
                # 
                if len(encoded['input_ids']) > 500:
                    encoded['input_ids'] = encoded['input_ids'][-500:]
                    encoded['token_type_ids'] = encoded['token_type_ids'][-500:]
                    encoded['attention_mask'] = encoded['attention_mask'][-500:]

                encoded['labels'] = label
                self.data.append(encoded)

        print("self.miss_entity_count", self.miss_entity_count)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)  

    def generate_sentence_from_context(self, entity: str):
        # Generate a sentence from the entity's context
        if entity not in self.kg:
            self.miss_entity_count += 1  # Increment the missing entity count
            return 'no context'

        sentence_list = []  
        
        # from_relations stores the subject entitys of giving entity
        # for instance, (entity_of_from_relations, predicate, giving entity)
        from_relations = self.kg[entity][config.FROM_KG]
        # Generate sentences from 'from_relations'
        for relation, objects in from_relations.items():
            template = self.prompt_templates[relation]
            objects_str = ', '.join(objects)
            sentence = template.format(subject_entity=objects_str, mask_token=entity)
            sentence_list.append(sentence)

        # to_relations stores the object entitys of giving entity
        to_relations = self.kg[entity][config.TO_KG]
        # for instance, (giving entity, predicate, entity_of_from_relations)
        # Generate sentences from 'to_relations'
        for relation, objects in to_relations.items():
            # template prompt
            template = self.prompt_templates[relation]
            objects_str = ', '.join(objects)
            sentence = template.format(subject_entity=entity, mask_token=objects_str)
            sentence_list.append(sentence)
        # join the sentence of triples into a sentence
        # for example, ['a relate to b','b relate to c'] into 'a relate to b and b relate to c'
        return ' and '.join(sentence_list)



def train():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_load_dir)
    bert_model: BertModel = transformers.BertForNextSentencePrediction.from_pretrained(
        args.model_load_dir, config=bert_config
    )
    # bert_tokenizer = transformers.AutoTokenizer.from_pretrained( args.model_load_dir)
    # tokenizer_dir = f'{config.RES_DIR}/tokenizer/bert'
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_load_dir)

    train_dataset = NSPDataset(
        tokenizer=bert_tokenizer,
        data_fn=args.train_fn,
        kg=kg,
        template_fn=args.template_fn,
    )
    print(f"train dataset size: {len(train_dataset)}")
    dev_dataset = NSPDataset(
        tokenizer=bert_tokenizer,
        data_fn=args.dev_fn,
        kg=kg,
        template_fn=args.template_fn,
    )

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
        # load_best_model_at_end=True,
    )

    trainer = transformers.Trainer(
        model=bert_model,
        train_dataset=train_dataset,
        args=training_args,
        eval_dataset=dev_dataset,
        tokenizer=bert_tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=args.model_best_dir)
    # dev_results = trainer.evaluate(dev_dataset)
    # print(f"dev results: ")
    # print(dev_results)


def evaluate():
    # load model from saved checkpoints in training stage
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)
    bert_model: BertModel = transformers.BertForNextSentencePrediction.from_pretrained(
        args.model_best_dir, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_best_dir)

    dev_dataset = NSPDataset(
        data_fn=args.test_fn,
        tokenizer=bert_tokenizer,
        kg=kg,
        template_fn=args.template_fn,
    )

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
    # obtain the raw confidence score of each triple
    predicts = trainer.predict(dev_dataset)

    # softmax to judgee easily
    predictions = util.softmax(predicts.predictions, axis=1)
    # [0.1,0.9]
    # the first column (predictions[:, 0]) is the confidence score of not correct and the second column (predictions[:, 1]) is the confidence score of correctness
    pre_score = predictions[:, 1]
    print(pre_score[:10])
    # if the confidence score of correctnesss over the threshold, set it to 1 elsewise 0
    pre = np.where(pre_score > 0.9995, 1, 0)

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
    # build knowledge graph from train set
    kg = util.KnowledgeGraph(args.train_fn)
    if "train" in args.mode:
        train()

    if "test" in args.mode:
        evaluate()
