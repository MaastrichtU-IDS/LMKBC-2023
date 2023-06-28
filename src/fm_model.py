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
from fm_dataset import MLMDataset
import config
from evaluate import evaluate
import util

task = "fill-mask"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

entity_set = set()
entity_fn = f"{config.DATA_DIR}/entity_set.json"
print(torch.cuda.is_available())


def train():
    if os.path.exists(best_dir):
        bert_config = transformers.AutoConfig.from_pretrained(best_dir)
        bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
            best_dir, config=bert_config
        )
        bert_tokenizer = transformers.AutoTokenizer.from_pretrained(best_dir)
    else:
        bert_config = transformers.AutoConfig.from_pretrained(args.pretrain_model_name)
        # print(bert_config)
        bert_model = transformers.AutoModelForMaskedLM.from_pretrained(
            args.pretrain_model_name, config=bert_config
        )
        bert_tokenizer: BertTokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.pretrain_model_name
        )
    bert_collator = transformers.DataCollatorForTokenClassification(
        tokenizer=bert_tokenizer,
        padding="max_length",
        max_length=config.FM_MAX_LENGTH,
    )
    train_dataset = MLMDataset(
        data_fn=args.train_fn, tokenizer=bert_tokenizer, template_fn=args.template_fn
    )
    print(f"train dataset size: {len(train_dataset)}")
    # we extend the mask window only for training set, in evaluation, we only really care about the object tokens themselves;
    # In these cases, we set ``extend_len'' to 0
    dev_dataset = MLMDataset(
        data_fn=args.dev_fn,
        tokenizer=bert_tokenizer,
        template_fn=args.template_fn,
    )
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
        # load_best_model_at_end=True,
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
    # compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(output_dir=best_dir)
    # bert_tokenizer.save_pretrained(args.bin_dir)
    dev_results = trainer.evaluate(dev_dataset)
    # trainer.model
    print(f"dev results: ")
    print(dev_results)


def test_pipeline():
    bert_config = transformers.AutoConfig.from_pretrained(best_dir)
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(
        best_dir, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(best_dir)

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
        # entity_set.update(row[KEY_OBJS])
    entity_escape_set = set()
    entity_in_set = set()
    for row in test_rows:
        objs = row[config.KEY_OBJS]
        for obj in objs:
            if obj not in entity_set:
                entity_escape_set.add(obj)
            else:
                entity_in_set.add(obj)
    print("escaped entity number: ", len(entity_escape_set))
    print("entity_in_set number: ", len(entity_in_set))
    # print("entity_escape_set", entity_escape_set)
    "1 2 3 [maxk] [mask] [mask] 4  5 6"
    # Run the model
    logger.info(f"Running the model...")

    outputs = pipe(prompts, batch_size=args.test_batch_size)
    logger.info(f"End the model...")
    results = []
    num_filtered = 0
    rel_thres_fn = f"{config.RES_PATH}/relation-threshold.json"
    if os.path.exists(rel_thres_fn):
        with open(rel_thres_fn, 'r') as f:
            rel_thres_dict = json.load(f)
    else:
        rel_thres_dict = dict()

    for row, output, prompt in zip(test_rows, outputs, prompts):
        objects_wikiid = []
        objects = []
        # print()
        # print(row[KEY_OBJS])
        for seq in output:
            # if row[KEY_REL] in printed_relation:
            # print("seq", seq)
            if row[config.KEY_REL] not in rel_thres_dict:
                print(f"{row[config.KEY_REL]} not in rel_thres_dict")
                rel_thres_dict[row[config.KEY_REL]] = args.threshold
            if seq["score"] > rel_thres_dict[row[config.KEY_REL]]:
                obj = seq["token_str"]
                if obj == config.EMPTY_TOKEN:
                    obj = ''
                # if obj not in entity_set:
                #     num_filtered += 1
                #     continue
                wikidata_id = util.disambiguation_baseline(obj)
                objects_wikiid.append(wikidata_id)
                objects.append(obj)

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
    output_fn = f"{config.OUTPUT_DIR}/{args.pretrain_model_name}_ressult.jsonl"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    with open(rel_thres_fn, 'w') as f:
        json.dump(rel_thres_dict, f)
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
    output_dir = f"{config.BIN_DIR}/{task}/{args.pretrain_model_name}"
    best_dir = f"{output_dir}/{args.bin_dir}"
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
