import math
import json
import argparse
import logging
import random
import numpy as np
import pandas as pd
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    DataCollatorForLanguageModeling,
    pipeline,
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    PreTrainedTokenizer,
)

import itertools
import config
from evaluate import evaluate
import util
from tqdm import tqdm

task = "pretrain_filled_mask"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

print(torch.cuda.is_available())

#os.environ['TRANSFORMERS_CACHE'] = 'cache/transformers/'



class PreSV_wiki_Dataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, data_fn) -> None:
        super().__init__()
        self.data = []
        train_data = util.file_read_json_line(data_fn)
        self.tokenizer = tokenizer
        max_length = 0 
        printable= True
        type_entity_fp =f'res/entity_for_pretrain.json'
        type_entity_dict = json.load(open(type_entity_fp))
        entity_type_dict = dict()
        for t,entity_list in type_entity_dict.items():
            for entity in entity_list:
                entity_type_dict[entity] = t
        print('entity_type_dict',len(entity_type_dict))
        for row in tqdm(train_data):
            entities = row['entities']
            sentence= row['sentence']
            tokens = [tokenizer.cls_token]+ row['tokens']+[tokenizer.sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if random.random()>0.5:
                origin_item = {
                "input_ids": input_ids,
                "labels": 1,
            }
                self.data.append(origin_item)
                continue
            
            entity_ids = tokenizer.convert_tokens_to_ids(entities)
            entity_ids = set(filter(lambda x: x!= tokenizer.unk_token_id, entity_ids))

            # print('entities', entities)
            # print('tokens',tokens)
            entity_index_ids = [i for i, v in enumerate(input_ids) if v in entity_ids]
            # print(entity_index_ids)
            if len(entity_index_ids) == 0:
                continue
            select_index = random.sample(entity_index_ids,1)[0]
            # entity_id = input_ids[select_index]
            token = input_ids[select_index]
            # print(select_index)
            # print("-------------------", token)
            select_token = tokenizer.convert_ids_to_tokens(input_ids[select_index])
            # print("-------------------",select_token)

            token_type = entity_type_dict[select_token]
            replace_token = random.sample(type_entity_dict[token_type],1)[0]
            input_ids_fake = input_ids[:]
            input_ids_fake[select_index] = tokenizer.convert_tokens_to_ids(replace_token)

    
            fake_item = {
                "input_ids": input_ids_fake,
                "labels": 0,
            }
            if printable:
                print("fake_item",fake_item)
                printable = False
            self.data.append(fake_item)

        random.shuffle(self.data)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



def train_sentence_validition():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_load_dir)
    bert_model: transformers.AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_load_dir, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)

    train_dataset = PreSV_wiki_Dataset(data_fn=args.train_fn, tokenizer=bert_tokenizer)
    bert_collator = util.DataCollatorKBC(
        tokenizer=bert_tokenizer,padding = False
    )
    # bert_model.forword()
    # collator = DataCollatorForLanguageModeling()
    print(f"train dataset size: {len(train_dataset)}")
    bert_model.resize_token_embeddings(len(bert_tokenizer))

    training_args = transformers.TrainingArguments(
        output_dir=args.model_save_dir,
        overwrite_output_dir=True,
        # evaluation_strategy='epoch',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=64,
        # eval_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.train_epoch,
        warmup_ratio=0.1,
        logging_dir=config.LOGGING_DIR,
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        # fp16=True,
        dataloader_num_workers=0,
        auto_find_batch_size=False,
        greater_is_better=False,
        # load_best_model_at_end=True,
        no_cuda=False,
    )

    trainer = transformers.Trainer(
        model=bert_model,
        data_collator=bert_collator,
        train_dataset=train_dataset,
        args=training_args,
        # eval_dataset=dev_dataset,
        tokenizer=bert_tokenizer,
    )
    # compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir=args.model_best_dir)
    print(f"best_ckpt_dir: ", args.model_best_dir)
    # print(dev_results)



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
        "--model_load_dir",
        type=str,
        help="HuggingFace model name (default: bert-base-cased)",
    )
    parser.add_argument(
        "--model_best_dir",
        type=str,
        help="HuggingFace model name (default: bert-base-cased)",
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
        "--mask_strategy",
        type=str,
        default="single",
        help="single,fold,random",
    )

    parser.add_argument(
        "--train_fn",
        type=str,
        required=True,
        help="CSV file containing train data for few-shot examples (required)",
    )
    parser.add_argument(
        "--train_epoch",
        type=float,
        default=10,
        help="CSV file containing train data for few-shot examples (required)",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for the model. (default:32)",
    )


    args = parser.parse_args()

    train_sentence_validition()
