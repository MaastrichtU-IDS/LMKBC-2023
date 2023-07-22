import json
import argparse
import logging
import numpy as np
import pandas as pd

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
os.environ['TRANSFORMERS_CACHE'] = config.TRANSFOER_CACHE_DIR
from evaluate import evaluate
import util

task = "token-align"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


print(torch.cuda.is_available())


class TADataset(Dataset):
    def __init__(
        self,
        tokenizer_origin: BertTokenizer,
        tokenizer_enhance: BertTokenizer,
        entity_set,
    ) -> None:
        super().__init__()
        self.data = []
        max_sentence_length = 0
        prompt = '{origin_word} is split into {tokens}'

        for entity in entity_set:
            tokens = tokenizer_origin.tokenize(entity)
            sentence_token = tokenizer_origin.tokenize('is equal to the following:')
            label_tokens = [entity] + sentence_token + tokens
            input_tokens = [tokenizer_origin.mask_token] + sentence_token + tokens

            label_ids = tokenizer_enhance.convert_tokens_to_ids(label_tokens)
            input_ids = tokenizer_enhance.convert_tokens_to_ids(input_tokens)
            attention_mask = [
                0 if v == tokenizer_enhance.mask_token else 1 for v in input_tokens
            ]
            # encoded_sentence = tokenizer.encode_plus(text=input_sentence)

            if len(input_ids) > max_sentence_length:
                max_sentence_length = len(input_ids)

            item = {
                "labels": label_ids,
                "input_ids": input_ids,
                # "attention_mask": attention_mask,
            }
            self.data.append(item)

        print("max_sentence_length", max_sentence_length)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def train():
    tokenizer_origin = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.bert_large_cased
    )
    tokenizer_enhance = transformers.AutoTokenizer.from_pretrained(
        config.TOKENIZER_PATH
    )

    bert_config = transformers.AutoConfig.from_pretrained(args.model_load_dir)
    # print(bert_config)
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_load_dir, config=bert_config
    )

    bert_collator = transformers.DataCollatorForTokenClassification(
        tokenizer=tokenizer_enhance,
        padding="max_length",
        max_length=config.TA_MAX_LENGTH,
    )
    with open(f'{config.TOKENIZER_PATH}/added_tokens.json') as f:
        entity_dict = json.load(f)

    entity_set_train = entity_dict.keys()
    train_dataset = TADataset(
        tokenizer_origin,
        tokenizer_enhance,
        entity_set_train,
    )
    print(f"train dataset size: {len(train_dataset)}")
    bert_model.resize_token_embeddings(len(tokenizer_enhance))

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
        fp16=True,
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
        tokenizer=tokenizer_enhance,
    )
    # compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir=args.model_best_dir)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Model with Question and Fill-Mask Prompts"
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

    parser.add_argument(
        "--train_epoch",
        type=int,
        default=10,
        help="CSV file containing train data for few-shot examples (required)",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for the model. (default:32)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=32,
        help="Batch size for the model. (default:32)",
    )

    args = parser.parse_args()
   
    train()
