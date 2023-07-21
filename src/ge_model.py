import csv
import json
import argparse
import logging
import random
import copy
from typing import List
import numpy as np
import pandas as pd
import requests

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import transformers
from transformers import pipeline, GPT2TokenizerFast
import os
import config
from evaluate import evaluate
from file_io import read_lm_kbc_jsonl
import util


from transformers import Trainer, TrainingArguments, AutoTokenizer


task = "text-generation"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GEDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_fn, template_fn) -> None:
        super().__init__()

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
            if len(object_entities) == 1 and object_entities[0] == config.EMPTY_STR:
                object_entities = [config.EMPTY_TOKEN]
            answers = ','.join(object_entities)
            instantiated_example = (
                prompt_template.format(subject_entity=subject) + f" {answers}"
            )
            encoded = tokenizer.encode_plus(instantiated_example)
            # print("encoded", encoded)
            if len(encoded['input_ids']) > max_sentence_length:
                max_sentence_length = len(encoded['input_ids'])
            self.data.append(encoded)

        print("max_sentence_length", max_sentence_length)
        print("max_obj_length", max_obj_length)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def create_prompt(
    subject_entity: str,
    relation: str,
    prompt_templates: dict,
    instantiated_templates,
    few_shot: int = 0,
) -> str:
    prompt_template = prompt_templates[relation]

    if few_shot > 0:
        random_examples = random.sample(
            instantiated_templates[relation], min(few_shot, len(instantiated_templates))
        )
    else:
        random_examples = []
    few_shot_examples = "\n".join(random_examples)
    prompt = (
        f"{few_shot_examples}\n{prompt_template.format(subject_entity=subject_entity)}"
    )
    return prompt


def train():

    print("load local model")
    bert_config = transformers.AutoConfig.from_pretrained(args.model_load_dir)
    model = transformers.OPTForCausalLM.from_pretrained(
        args.model_load_dir, config=bert_config
    )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path=best_dir, padding_side='left'
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_load_dir,
        padding_side='left',
        padding=True,
    )
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        # max_length=config.GE_MAX_LENGTH,
    )

    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

    # dataset = transformers.LineByLineTextDataset(
    #     tokenizer=tokenizer,
    #     file_path="./text.txt",
    #     block_size=128,
    # )

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )

    train_dataset = GEDataset(
        data_fn=args.train_fn, tokenizer=tokenizer, template_fn=args.template_fn
    )
    print(f"train dataset size: {len(train_dataset)}")
    # we extend the mask window only for training set, in evaluation, we only really care about the object tokens themselves;
    # In these cases, we set ``extend_len'' to 0
    dev_dataset = GEDataset(
        data_fn=args.dev_fn,
        tokenizer=tokenizer,
        template_fn=args.template_fn,
    )

    # tokenizer.save(best_dir)
    # model.resize_token_embeddings(len(tokenizer))
    training_args = TrainingArguments(
        output_dir=args.model_save_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.train_epoch,
        per_device_train_batch_size=args.train_batch_size,
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=args.learning_rate,
        # load_best_model_at_end=True,
        warmup_ratio=0.1,
        logging_strategy='epoch'
        # evaluation_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
    # compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir=args.model_best_dir)
    # bert_tokenizer.save_pretrained(args.bin_dir)
    dev_results = trainer.evaluate(dev_dataset)
    # trainer.model
    print(f"dev results: ")
    print(dev_results)


def test_pipeline():


    print("load local model")
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_best_dir, config=bert_config
    )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     best_dir, padding_side='left'
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.opt_350m, padding_side='left'
    )

    instantiated_templates = []
    relation_instantiated = dict()

    # logger.info(f"Reading train data from \"{args.train_data}\"...")
    train_data = util.file_read_json_line(args.train_fn)
    prompt_templates = util.file_read_prompt(args.template_fn)
    logger.info("Instantiating templates with train data...")
    for row in train_data:
        relation = row['Relation']
        prompt_template = prompt_templates[relation]
        object_entities = row['ObjectEntities']
        answers = ', '.join(object_entities)
        instantiated_example = (
            prompt_template.format(subject_entity=row["SubjectEntity"]) + f" {answers}"
        )
        instantiated_templates.append(instantiated_example)
        if relation not in relation_instantiated:
            relation_instantiated[relation] = []
        relation_instantiated[relation].append(instantiated_example)

    pipe = pipeline(task, model=model, tokenizer=tokenizer, device=args.gpu)
    logger.info(f"Creating prompts...")
    test_rows = util.file_read_json_line(args.test_fn)
    prompts = [
        create_prompt(
            subject_entity=row["SubjectEntity"],
            relation=row["Relation"],
            prompt_templates=prompt_templates,
            instantiated_templates=relation_instantiated,
            few_shot=args.few_shot,
        )
        for row in test_rows
    ]

    logger.info(f"Running the model...")

    outputs = tqdm(
        pipe(prompts, batch_size=args.test_batch_size, max_length=config.GE_MAX_LENGTH),
        total=len(prompts),
    )
    logger.info(f"End the model...")
    results = []
    for row, output, prompt in tqdm(zip(test_rows, outputs, prompts)):
        objects_wikiid = []
        # print("output", output)
        qa_answer = output[0]['generated_text'].split(prompt)[-1].strip()
        objects = qa_answer.split(",")
        objects = [answer.strip() for answer in objects]
        objects = list(set(objects))
        if config.EMPTY_TOKEN in objects:
            objects = [config.EMPTY_STR]
        # for entity in objects:
        #     wikidata_id = util.disambiguation_baseline(entity)
        #     objects_wikiid.append(wikidata_id)
        is_in = []
        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": objects_wikiid,
            "ObjectEntities": objects,
            "Relation": row["Relation"],
            "ObjectExists": is_in,
        }
        results.append(result_row)
    # Save the results
    # output_fn = f"{config.OUTPUT_DIR}/{args.pretrain_model_name}_ressult.jsonl"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    logger.info(f"Saving the results to \"{args.output}\"...")

    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Start Evaluate ...")
    evaluate(args.output, args.test_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Model with Question and Fill-Mask Prompts"
    )
    parser.add_argument(
        "-m",
        "--pretrain_model_name",
        type=str,
        default="bert-base-cased",
        help="HuggingFace model name (default: bert-base-cased)",
    )
    parser.add_argument(
        "--bin_dir",
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
        "--test_batch_size",
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
    parser.add_argument(
        "--few_shot",
        type=int,
        default=3,
        help="Batch size for the model. (default:32)",
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


    args = parser.parse_args()

    if "train" in args.mode:
        train()

    if "test" in args.mode:
        test_pipeline()

    if "evaulate" in args.mode:
        evaluate(args.output, args.test_fn)
