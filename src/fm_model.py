import json
import argparse
import logging

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

import torch
from torch.utils.data import Dataset
import transformers
from transformers import pipeline, BertTokenizerFast, BertModel
import os

from transformers.utils import PaddingStrategy
import config
from evaluate import evaluate
import util
import tqdm

task = "fill-mask"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

print(torch.cuda.is_available())


class MLMDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizerFast, data_fn, template_fn) -> None:
        super().__init__()
        self.data = []
        train_data = util.file_read_json_line(data_fn)
        prompt_templates = util.file_read_prompt(template_fn)
        for row in train_data:
            relation = row['Relation']
            prompt_template = prompt_templates[relation]
            object_entities = row['ObjectEntities']
            subject = row["SubjectEntity"]
            for obj in object_entities:
                if obj == '':
                    obj = config.EMPTY_TOKEN
                input_sentence = prompt_template.format(
                    subject_entity=subject, mask_token=tokenizer.mask_token
                )
                obj_id = tokenizer.convert_tokens_to_ids(obj)
                input_ids, attention_mask = util.tokenize_sentence(
                    tokenizer, input_sentence
                )
                label_ids = [
                    obj_id if t == tokenizer.mask_token_id else -100 for t in input_ids
                ]

                item = {
                    "labels": label_ids,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                self.data.append(item)

        print(self.data[0])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def train():
    # if os.path.exists(args.model_load_dir):
    # print(f"using existing model {args.model_load_dir}")
    bert_config = transformers.AutoConfig.from_pretrained(args.model_load_dir)
    bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_load_dir, config=bert_config
    )
    # else:
    #     print(f"using huggingface  model {args.model_load_dir}")
    #     bert_config = transformers.AutoConfig.from_pretrained(config.bert_base_cased)
    #     bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
    #         config.bert_base_cased, config=bert_config
    #     )

    is_train = os.path.isdir(args.model_load_dir)

    train_dataset = MLMDataset(
        data_fn=args.train_fn, tokenizer=bert_tokenizer, template_fn=args.template_fn
    )
    bert_collator = util.DataCollatorKBC(
        tokenizer=bert_tokenizer,
    )
    dev_dataset = MLMDataset(
        data_fn=args.dev_fn,
        tokenizer=bert_tokenizer,
        template_fn=args.template_fn,
    )
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
        data_collator=bert_collator,
        train_dataset=train_dataset,
        args=training_args,
        eval_dataset=dev_dataset,
        tokenizer=bert_tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=args.model_best_dir)
    print(f"model_best_dir: ", args.model_best_dir)


def test_pipeline():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)
    bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_best_dir, config=bert_config
    )

    pipe = pipeline(
        task=task,
        model=bert_model,
        tokenizer=bert_tokenizer,
        top_k=args.top_k,
        device=args.gpu,
    )
    prompt_templates = util.file_read_prompt(args.template_fn)
    test_rows = util.file_read_json_line(args.test_fn)
    prompts = []
    for row in test_rows:
        prompt = prompt_templates[row["Relation"]].format(
            subject_entity=row["SubjectEntity"],
            mask_token=bert_tokenizer.mask_token,
        )
        prompts.append(prompt)

    logger.info(f"Running the model...")

    outputs = pipe(prompts, batch_size=args.train_batch_size * 4)
    logger.info(f"End the model...")
    results = []
    num_filtered = 0
    rel_thres_fn = f"{config.RES_DIR}/relation-threshold.json"
    if os.path.exists(rel_thres_fn):
        with open(rel_thres_fn, 'r') as f:
            rel_thres_dict = json.load(f)
    else:
        rel_thres_dict = dict()

    for row, output, prompt in zip(test_rows, outputs, prompts):
        objects_wikiid = []
        objects = []
        for seq in output:
            if row[config.KEY_REL] not in rel_thres_dict:
                print(f"{row[config.KEY_REL]} not in rel_thres_dict")
                rel_thres_dict[row[config.KEY_REL]] = args.threshold
            if seq["score"] > rel_thres_dict[row[config.KEY_REL]]:
                obj = seq["token_str"]
                if obj == config.EMPTY_TOKEN:
                    objects_wikiid.append(config.EMPTY_STR)
                    objects = [config.EMPTY_STR]
                    break
                else:
                    wikidata_id = util.disambiguation_baseline(obj)
                    objects_wikiid.append(wikidata_id)
                objects.append(obj)
        if config.EMPTY_STR in objects:
            objects = [config.EMPTY_STR]

        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": objects_wikiid,
            "ObjectEntities": objects,
            "Relation": row["Relation"],
        }
        results.append(result_row)
    print("filtered entity number: ", num_filtered)
    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    util.file_write_json_line(args.output, results)
    logger.info(f"Start Evaluate ...")
    evaluate(args.output, args.test_fn)


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
    tokenizer_dir = f'{config.RES_DIR}/tokenizer/bert'
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)

    if "train" in args.mode:
        train()

    if "test" in args.mode:
        test_pipeline()
