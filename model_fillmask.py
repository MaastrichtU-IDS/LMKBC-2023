import csv
import json
import argparse
import logging
import random
import copy
import pandas as pd
import requests

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline, BertTokenizer
import os
import config
from evaluate import combine_scores_per_relation, evaluate_per_sr_pair
from file_io import read_lm_kbc_jsonl
import util

task = "fill-mask"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MLMDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, data_fn, template_fn) -> None:
        super().__init__()
        self.data = []
        self.label = []
        with open(data_fn, "r") as file:
            train_data = [json.loads(line) for line in file]
        with open(template_fn, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            prompt_templates = {
                row['Relation']: row['PromptTemplate'] for row in reader
            }

        for row in train_data:
            relation = row['Relation']
            prompt_template = prompt_templates[relation]
            object_entities = row['ObjectEntities']
            for obj in object_entities:
                label = prompt_template.format(
                    subject_entity=row["SubjectEntity"], mask_token=obj
                )
                input_token = prompt_template.format(
                    subject_entity=row["SubjectEntity"], mask_token=tokenizer.mask_token
                )

                label_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label))
                input_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(input_token)
                )
                item = {
                    "labels": label_ids,
                    "input_ids": input_ids,
                }
                self.data.append(item)
        print(self.data[0])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def train(
    # model_name: str,
    # train_fns: str,
    # dev_fn: str,
    # test_fn: str,
    # template_fn: str,
):
    bert_config = transformers.AutoConfig.from_pretrained(args.pretrain_model_name)
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(
        args.pretrain_model_name, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.pretrain_model_name
    )
    bert_collator = transformers.DataCollatorForTokenClassification(
        tokenizer=bert_tokenizer, padding="max_length", max_length=128
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
    test_dataset = MLMDataset(
        data_fn=args.test_fn,
        tokenizer=bert_tokenizer,
        template_fn=args.template_fn,
    )

    training_args = transformers.TrainingArguments(
        output_dir=args.bin_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=8,
        eval_delay=0,
        learning_rate=4e-5,
        weight_decay=0,
        num_train_epochs=args.train_epoch,
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
        log_level='debug',
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
    # compute_metrics=compute_metrics)

    trainer.train()
    trainer.save_model(output_dir=os.path.join(config.BIN_DIR, 'best_ckpt'))
    dev_results = trainer.evaluate(dev_dataset)
    # trainer.model
    print(f"dev results: ")
    print(dev_results)

    return


def test(
    test_fn: str,
    bin_dir: str,
    template_fn: str,
):
    bert_config = transformers.AutoConfig.from_pretrained(bin_dir)
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(
        bin_dir, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(bin_dir)
    bert_collator = transformers.DataCollatorForTokenClassification(
        tokenizer=bert_tokenizer, padding="max_length", max_length=128
    )

    training_args = transformers.TrainingArguments(
        output_dir=config.BIN_DIR,
        overwrite_output_dir=False,
        evaluation_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        per_gpu_eval_batch_size=64,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=8,
        eval_delay=0,
        learning_rate=4e-4,
        weight_decay=0,
        num_train_epochs=args.train_epoch,
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
        log_level='debug',
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
        tokenizer=bert_tokenizer,
        args=training_args,
    )

    inference_dataset = MLMDataset(
        data_fn=test_fn, tokenizer=bert_tokenizer, template_fn=template_fn
    )
    print(f"eval dataset size: {len(inference_dataset)}")

    inference_results = trainer.predict(inference_dataset)
    print(inference_results.keys())


def test_pipeline(
    # test_fn: str,
    # bin_dir: str,
    # template_fn: str,
    # output,
):
    bert_config = transformers.AutoConfig.from_pretrained(args.bin_dir)
    bert_model = transformers.AutoModelForMaskedLM.from_pretrained(
        args.bin_dir, config=bert_config
    )
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(args.bin_dir)
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
    input_rows = [json.loads(line) for line in open(args.test_fn, "r")]
    prompts = [
        util.create_prompt(
            subject_entity=row["SubjectEntity"],
            relation=row["Relation"],
            prompt_templates=prompt_templates,
            instantiated_templates=None,
            tokenizer=bert_tokenizer,
            few_shot=args.few_shot,
            task=task,
        )
        for row in input_rows
    ]

    # Run the model
    logger.info(f"Running the model...")
    if task == 'fill-mask':
        outputs = pipe(prompts, batch_size=args.test_batch_size)
    else:
        outputs = pipe(prompts, batch_size=args.test_batch_size, max_length=512)

    results = []
    for row, output, prompt in zip(input_rows, outputs, prompts):
        object_entities_with_wikidata_id = []
        if task == "fill-mask":
            for seq in output:
                if seq["score"] > args.threshold:
                    wikidata_id = util.disambiguation_baseline(seq["token_str"])
                    object_entities_with_wikidata_id.append(wikidata_id)
        else:
            # Remove the original prompt from the generated text
            qa_answer = output[0]['generated_text'].split(prompt)[-1].strip()
            qa_entities = qa_answer.split(", ")
            for entity in qa_entities:
                wikidata_id = util.disambiguation_baseline(entity)
                object_entities_with_wikidata_id.append(wikidata_id)

        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": object_entities_with_wikidata_id,
            "Relation": row["Relation"],
        }
        results.append(result_row)

    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    evaluate()


def evaluate():
    pred_rows = read_lm_kbc_jsonl(args.output)
    gt_rows = read_lm_kbc_jsonl(args.test_fn)

    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["*** Average ***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
    }

    print(pd.DataFrame(scores_per_relation).transpose().round(3))


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
        test_pipeline()
