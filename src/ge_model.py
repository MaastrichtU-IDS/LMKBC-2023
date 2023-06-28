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
from transformers import pipeline, BertModel, GPT2TokenizerFast
import os
from ge_dataset import GEDataset
import config
from evaluate import evaluate
from file_io import read_lm_kbc_jsonl
import util


from transformers import (
    Trainer,
    TrainingArguments,
)


task = "text-generation"
KEY_OBJS = "ObjectEntities"
KEY_REL = "Relation"
KEY_SUB = "SubjectEntity"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

entity_set = set()
entity_fn = f"{config.DATA_DIR}/entity_set.json"
print(torch.cuda.is_available())


def create_prompt(
    subject_entity: str,
    relation: str,
    prompt_templates: dict,
    instantiated_templates,
    tokenizer,
    few_shot: int = 0,
    task: str = "fill-mask",
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
    output_dir = f"{config.BIN_DIR}/{task}/{args.pretrain_model_name}"
    best_dir = f"{output_dir}/{args.bin_dir}"
    if os.path.exists(best_dir):
        print("load local model")
        bert_config = transformers.AutoConfig.from_pretrained(best_dir)
        model: BertModel = transformers.AutoModelForCausalLM.from_pretrained(
            best_dir, config=bert_config
        )
        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=best_dir, padding_side='left'
        # )
    else:
        print("load remote model")
        bert_config = transformers.AutoConfig.from_pretrained(args.pretrain_model_name)
        # print(bert_config)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.pretrain_model_name, config=bert_config
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrain_model_name, padding_side='left'
    )
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        # max_length=config.GE_MAX_LENGTH,
    )

    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

    # dataset = LineByLineTextDataset(
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
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.train_epoch,
        per_device_train_batch_size=args.train_batch_size,
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=args.learning_rate,
        # load_best_model_at_end=True,
        warmup_ratio=0.1,
        evaluation_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
    # compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir=best_dir)
    # bert_tokenizer.save_pretrained(args.bin_dir)
    dev_results = trainer.evaluate(dev_dataset)
    # trainer.model
    print(f"dev results: ")
    print(dev_results)


def test_pipeline():
    model_dir = f"{config.BIN_DIR}/{task}/{args.pretrain_model_name}"
    best_dir = f"{model_dir}/{args.bin_dir}"
    print(model_dir)
    if os.path.exists(best_dir):
        print("load local model")
        bert_config = transformers.AutoConfig.from_pretrained(best_dir)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            best_dir, config=bert_config
        )
        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     best_dir, padding_side='left'
        # )
    else:
        print("load remote model")
        bert_config = transformers.AutoConfig.from_pretrained(args.pretrain_model_name)
        # print(bert_config)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.pretrain_model_name, config=bert_config
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.pretrain_model_name, padding_side='left'
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
            tokenizer=tokenizer,
            few_shot=args.few_shot,
            task=task,
        )
        for row in test_rows
    ]

    entity_escape_set = set()
    entity_in_set = set()
    for row in test_rows:
        objs = row[KEY_OBJS]
        for obj in objs:
            if obj in entity_set or obj in tokenizer.vocab:
                entity_in_set.add(obj)
            else:
                entity_escape_set.add(obj)

    print("escaped entity number: ", len(entity_escape_set))
    print("entity_in_set", len(entity_in_set))
    "1 2 3 [maxk] [mask] [mask] 4  5 6"
    # Run the model
    logger.info(f"Running the model...")

    outputs = tqdm(
        pipe(prompts, batch_size=args.test_batch_size, max_length=config.GE_MAX_LENGTH)
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
        # for entity in objects:
        #     wikidata_id = util.disambiguation_baseline(entity)
        #     objects_wikiid.append(wikidata_id)
        is_in = []
        for obj in objects:
            if obj in entity_set or obj in tokenizer.vocab:
                is_in.append(1)
            else:
                is_in.append(0)

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


def build_entity_set(fn):
    with open(args.train_fn, "r") as file:
        for line in file:
            line_json = json.loads(line)
            object_entities = line_json['ObjectEntities']
            subject = line_json["SubjectEntity"]

            entity_set.add(subject)
            entity_set.update(object_entities)


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

    args = parser.parse_args()

    if not os.path.exists(entity_fn):
        build_entity_set(args.train_fn)
        build_entity_set(args.dev_fn)
        with open(entity_fn, 'w') as f:
            json.dump(list(entity_set), f)
    else:
        with open(entity_fn, 'r') as f:
            entity_set = set(json.load(f))

    if "train" in args.mode:
        train()

    if "test" in args.mode:
        test_pipeline()
