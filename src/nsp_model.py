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
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
    BertTokenizer,
    BertForMaskedLM,
    BertModel,
)
import os
from nsp_dataset import NSPDataset
import config
from evaluate import combine_scores_per_relation, evaluate_per_sr_pair
from file_io import read_lm_kbc_jsonl
import util

import sklearn.metrics as metrics

task = 'nsp'
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train():
    output_dir = f"{config.BIN_DIR}/{task}/{args.pretrain_model_name}"
    best_dir = f"{output_dir}/{args.bin_dir}"
    if os.path.exists(best_dir):
        bert_config = transformers.AutoConfig.from_pretrained(best_dir)
        bert_model: BertModel = (
            transformers.BertForNextSentencePrediction.from_pretrained(
                best_dir, config=bert_config
            )
        )

        bert_tokenizer = transformers.AutoTokenizer.from_pretrained(best_dir)
    else:
        bert_config = transformers.AutoConfig.from_pretrained(args.pretrain_model_name)
        # print(bert_config)
        bert_model = transformers.BertForNextSentencePrediction.from_pretrained(
            args.pretrain_model_name, config=bert_config
        )
        bert_tokenizer: BertTokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.pretrain_model_name
        )
    # bert_collator = transformers.DataCollatorForLanguageModeling(
    #     tokenizer=bert_tokenizer,
    #     padding="max_length",
    #     max_length=config.MAX_LENGTH,
    # )
    train_dataset = NSPDataset(data_fn=args.train_fn, tokenizer=bert_tokenizer)
    print(f"train dataset size: {len(train_dataset)}")
    # we extend the mask window only for training set, in evaluation, we only really care about the object tokens themselves;
    # In these cases, we set ``extend_len'' to 0
    dev_dataset = NSPDataset(data_fn=args.dev_fn, tokenizer=bert_tokenizer)
    bert_model.resize_token_embeddings(len(bert_tokenizer))

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
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
        # data_collator=bert_collator,
        train_dataset=train_dataset,
        args=training_args,
        eval_dataset=dev_dataset,
        tokenizer=bert_tokenizer,
    )
    # compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir=best_dir)
    # bert_tokenizer.save_pretrained(args.bin_dir)
    dev_results = trainer.evaluate(dev_dataset)
    # trainer.model
    print(f"dev results: ")
    print(dev_results)


def predict():
    output_dir = f"{config.BIN_DIR}/{task}/{args.pretrain_model_name}"
    best_dir = f"{output_dir}/{args.bin_dir}"

    bert_config = transformers.AutoConfig.from_pretrained(best_dir)
    bert_model: BertModel = transformers.BertForNextSentencePrediction.from_pretrained(
        best_dir, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(best_dir)

    test_dataset = NSPDataset(data_fn=args.test_fn, tokenizer=bert_tokenizer)
    bert_model.resize_token_embeddings(len(bert_tokenizer))

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
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


def test_pipeline():
    model_dir = f"{config.BIN_DIR}/{args.pretrain_model_name}"
    best_dir = f"{model_dir}/{args.bin_dir}"
    bert_config = transformers.AutoConfig.from_pretrained(best_dir)
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(
        best_dir, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(best_dir)
    bert_collator = transformers.DataCollatorForTokenClassification(
        tokenizer=bert_tokenizer, padding="max_length", max_length=128
    )

    pipe = pipeline(
        task=task,
        model=bert_model,
        tokenizer=bert_tokenizer,
        top_k=args.top_k,
        device=args.gpu,
    )
    prompt_templates = util.read_prompt_templates_from_csv(args.template_fn)
    test_rows = [json.loads(line) for line in open(args.test_fn, "r")]
    prompts = []
    for row in test_rows:
        prompt = prompt_templates[row["Relation"]].format(
            subject_entity=row["SubjectEntity"],
            mask_token=bert_tokenizer.mask_token,
        )
        prompts.append(prompt)
        # entity_set.update(row[KEY_OBJS])
    num_entity_escape = 0
    for row in test_rows:
        objs = row[KEY_OBJS]
        for obj in objs:
            if obj not in entity_set:
                num_entity_escape += 1
    print("escaped entity number: ", num_entity_escape)
    "1 2 3 [maxk] [mask] [mask] 4  5 6"
    # Run the model
    logger.info(f"Running the model...")

    outputs = pipe(prompts, batch_size=args.test_batch_size)

    results = []
    num_filtered = 0
    printed_relation = {"person-has-spouse"}
    rel_thres_fn = f"{config.RES_PATH}/relation-threshold.json"
    with open(rel_thres_fn, 'r') as f:
        rel_thres_dict = json.load(f)
    for row, output, prompt in zip(test_rows, outputs, prompts):
        objects_wikiid = []
        # print()
        # print(row[KEY_OBJS])
        for seq in output:
            # if row[KEY_REL] in printed_relation:
            # print("seq", seq)
            if seq["score"] > rel_thres_dict[row[KEY_REL]]:
                obj = seq["token_str"]
                if obj == config.EMPTY_TOKEN:
                    obj = ''
                if obj not in entity_set:
                    num_filtered += 1
                    continue
                wikidata_id = util.disambiguation_baseline(obj)
                objects_wikiid.append(wikidata_id)

        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": objects_wikiid,
            "Relation": row["Relation"],
        }
        results.append(result_row)
    print("filtered entity number: ", num_filtered)
    # Save the results
    output_fn = f"{config.OUTPUT_DIR}/{args.pretrain_model_name}_ressult.jsonl"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    logger.info(f"Saving the results to \"{output_fn}\"...")
    with open(output_fn, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    evaluate()


def evaluate():
    output_fn = f"{config.OUTPUT_DIR}/{args.pretrain_model_name}_ressult.jsonl"
    pred_rows = read_lm_kbc_jsonl(output_fn)
    gt_rows = read_lm_kbc_jsonl(args.test_fn)

    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["*** Average ***---"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
    }
    scores_per_relation_pd = pd.DataFrame(scores_per_relation)

    print(scores_per_relation_pd.transpose().round(3))
    # average_pd = scores_per_relation_pd.mean(axis=1)
    # print(average_pd.round(3))


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
        "-k",
        "--top_k",
        type=int,
        default=5,
        help="Top k prompt outputs (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
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
        "-f",
        "--few_shot",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)",
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

    args = parser.parse_args()

    if "train" in args.mode:
        train()

    if "test" in args.mode:
        predict()
