import math
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

with open(config.TOKENIZER_PATH+"/added_tokens.json") as f:
    additional_token_dict = json.load(f)

class PreFM_wiki_Dataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, data_fn) -> None:
        super().__init__()
        self.data = []
        train_data = util.file_read_json_line(data_fn)
        self.tokenizer = tokenizer
        max_length = 0 
        printable= True
        for row in tqdm(train_data):
            entities = row['entities']
            sentence= row['sentence']
            tokens = [tokenizer.cls_token]+ row['tokens']+[tokenizer.sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            entity_ids = tokenizer.convert_tokens_to_ids(entities)
            entity_ids = set(filter(lambda x: x!= tokenizer.unk_token_id, entity_ids))
            entity_set_ids = entity_ids
            # generate masking-combination, for example, a sentence contains three entities, e.g. (a,b,c). We can select one,multiple or all of them, that is (a),(b),(c),(a,b),(a,c),etc. Different permutation scheme may provides different performance
            entity_index_ids = [i for i, v in enumerate(input_ids) if v in entity_set_ids]
            if len(entity_index_ids) == 0:
                continue
            random.shuffle(entity_index_ids)
            if 'fold' in args.mask_strategy:
                select_index_list = self._mask_fold(entity_index_ids)
            elif 'single' in args.mask_strategy:
                select_index_list = self._mask_single(entity_index_ids)
            elif "random" in args.mask_strategy:
                select_index_list =[random.sample(range(1,len(input_ids)-1), max(1,int(0.15*len(input_ids))))]
            else:
                raise Exception("no mask strategy")

            if printable and False:
                print("exists",entities)
                print("input_tokens",tokenizer.convert_ids_to_tokens(input_ids))
            if len(input_ids) > max_length:
                max_length = len(input_ids)
                #print("row",row)
         
            for mask_index in select_index_list:
                if len(mask_index) == 0:
                    continue
                    #raise Exception("mask index is zero")
                # in label id sequences, only the loss of  masked tokens will be feedback to update the model, the loss of other tokens will be discard.
                # label_ids = [-100]*len(input_ids)
                label_ids = [v if i in mask_index else -100  for i, v in enumerate(input_ids)]
                #  in input id sequences, the weight of masked tokens will  be zero. That means the vector of masked tokens in input_ids will not be considered in predicting the mask entities in label_ids.
                attention_mask =[0 if i in mask_index else 1  for i, v in enumerate(input_ids)]
                # replace the id of entities in input_ids with the mask_token_id
                input_ids_t = [tokenizer.mask_token_id if i in mask_index else v  for i, v in enumerate(input_ids)]
                item = {
                    "input_ids": input_ids_t,
                    "labels": label_ids,
                    "attention_mask": attention_mask,
                }
                if printable:
                    print("item",item)
                    printable = False
                self.data.append(item)

        random.shuffle(self.data)
        print("max_length",max_length)

    def _mask_fold(self, entity_index_ids):
        split_size = min(7, len(entity_index_ids))
        chunk_size = math.ceil(len(entity_index_ids) / split_size )
        select_index_list = [entity_index_ids[i*chunk_size:(i+1)*chunk_size] for i in range(split_size)]
        return select_index_list
    

    def _mask_single(self, entity_index_ids):

        mask_size = max(1, len(entity_index_ids)//7)
        # mask_size = 1
        select_index_list =[random.sample(entity_index_ids, mask_size)]
        return select_index_list
    

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def train_fm():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_load_dir)
    bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_load_dir, config=bert_config
    )

    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
    if not os.path.isdir( args.model_load_dir) and args.token_recode:
        print("repair token embedding")
        origin_tokenizer = transformers.AutoTokenizer.from_pretrained(config.bert_base_cased)
        util.token_layer(bert_model, additional_token_dict, bert_tokenizer, origin_tokenizer)
    train_dataset = PreFM_wiki_Dataset(data_fn=args.train_fn, tokenizer=bert_tokenizer)
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

    parser.add_argument(
        "--token_recode",
        type=util.str2bool,
        default=False,
        help="Batch size for the model. (default:32)",
    )

    


    args = parser.parse_args()

    train_fm()