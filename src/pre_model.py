import json
import argparse
import logging
import random
import numpy as np
import pandas as pd

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
import os
import itertools
import config
from evaluate import evaluate
import util
import tqdm

task = "pretrain_filled_mask"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

print(torch.cuda.is_available())


class PREMLMDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, data_fn) -> None:
        super().__init__()
        self.data = []
        train_data = util.file_read_json_line(data_fn)
        self.tokenizer = tokenizer
        for row in train_data:
            exists = row['exists']
            input_tokens = [tokenizer.cls_token] + row['tokens'] + [tokenizer.sep_token]
            exists_ids = tokenizer.convert_tokens_to_ids(exists)
            # exists_ids_thin = random.sample(exists_ids, min(10, len(exists_ids)))
            # select_ids_list = list(
            #     itertools.permutations(
            #         exists_ids_thin, max(1, int(len(exists_ids_thin) * 0.6))
            #     )
            # )
            # select_ids_list = random.sample(
            #     select_ids_list, min(1, len(select_ids_list))
            # )
            select_ids_list = [exists_ids]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            for select_ids in select_ids_list:
                label_ids = [x if x in select_ids else -100 for x in input_ids]
                attention_mask = [0 if x in select_ids else 1 for x in input_ids]
                input_ids_t = [
                    tokenizer.mask_token_id if x in select_ids else x for x in input_ids
                ]
                item = {
                    "input_ids": input_ids_t,
                    "labels": label_ids,
                    "attention_mask": attention_mask,
                }
                self.data.append(item)

        random.shuffle(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def train():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_load_dir)
    bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_load_dir, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)

    train_dataset = PREMLMDataset(data_fn=args.train_fn, tokenizer=bert_tokenizer)
    bert_collator = util.DataCollatorKBC(
        tokenizer=bert_tokenizer,
    )
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
        # eval_dataset=dev_dataset,
        tokenizer=bert_tokenizer,
    )
    # compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir=best_ckpt_dir)
    print(f"best_ckpt_dir: ", best_ckpt_dir)
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
    best_ckpt_dir = args.model_save_dir + "/best_ckpt"
    tokenizer_dir = f'{config.RES_DIR}/tokenizer/bert'

    if "train" in args.mode:
        train()
