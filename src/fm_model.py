import csv
import json
import argparse
import logging
import random
import pandas as pd

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import transformers
from transformers import pipeline, BertTokenizerFast, BertModel
import os


import config
os.environ['TRANSFORMERS_CACHE'] = config.TRANSFOER_CACHE_DIR
import evaluate
# from evaluate import evaluate
import util

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

        # Read the training data from a JSON file
        train_data = util.file_read_json_line(data_fn)

        # Read the prompt templates from a file
        prompt_templates = util.file_read_prompt(template_fn)

        # Iterate over each row in the training data
        for row in train_data:
            relation = row['Relation']
            prompt_template = prompt_templates[relation]
            object_entities = row['ObjectEntities']
            subject = row["SubjectEntity"]

            # Iterate over each object entity
            for obj in object_entities:
                if obj == '':
                    obj = config.EMPTY_TOKEN

                # Create an input sentence by formatting the prompt template
                input_sentence = prompt_template.format(
                    subject_entity=subject, mask_token=tokenizer.mask_token
                )

                # Convert the object entity to its corresponding ID
                obj_id = tokenizer.convert_tokens_to_ids(obj)

                # Tokenize the input sentence using the tokenizer
                input_ids, attention_mask = util.tokenize_sentence(
                    tokenizer, input_sentence
                )

                # Create label IDs where the masked token corresponds to the object ID
                label_ids = [
                    obj_id if t == tokenizer.mask_token_id else -100 for t in input_ids
                ]

                item = {
                    "labels": label_ids,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                self.data.append(item)
        random.shuffle(self.data)
        print(self.data[0])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def train():
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

    # is_train can be used to resume train process in formal training
    is_train = os.path.isdir(args.model_load_dir)
    # Create the training dataset using the MLMDataset class, providing the train data file, tokenizer, and template file
    train_dataset = MLMDataset(
        data_fn=args.train_fn, tokenizer=bert_tokenizer, template_fn=args.template_fn
    )

    # Create a data collator for batching and padding the data during training
    bert_collator = util.DataCollatorKBC(
        tokenizer=bert_tokenizer,
    )

    # Create the development dataset using the MLMDataset class, providing the dev data file, tokenizer, and template file
    dev_dataset = MLMDataset(
        data_fn=args.dev_fn,
        tokenizer=bert_tokenizer,
        template_fn=args.template_fn,
    )

    # Resize the token embeddings of the BERT model to match the tokenizer's vocabulary size
    bert_model.resize_token_embeddings(len(bert_tokenizer))

    # Set up the training arguments for the model
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

    # Create a trainer object for training the model
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
    # Load the configuration for the trained BERT model from the specified directory
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)

    # Load the trained BERT model for masked language modeling
    bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_best_dir, config=bert_config
    )

    # Create a pipeline for the specified task using the loaded BERT model and tokenizer
    pipe = pipeline(
        task=task,
        model=bert_model,
        tokenizer=bert_tokenizer,
        top_k=args.top_k,
        device=args.gpu,
    )

    # Read the prompt templates from the specified file
    prompt_templates = util.file_read_prompt(args.template_fn)

    # Read the test data rows from the specified file
    test_rows = util.file_read_json_line(args.test_fn)

    prompts = []
    for row in test_rows:
        # Generate a prompt for each test row using the corresponding template
        prompt = prompt_templates[row["Relation"]].format(
            subject_entity=row["SubjectEntity"],
            mask_token=bert_tokenizer.mask_token,
        )
        prompts.append(prompt)

    logger.info(f"Running the model...")

    # Run the model on the generated prompts in batches
    outputs = pipe(prompts, batch_size=args.train_batch_size * 4)

    logger.info(f"End the model...")

    results = []
    for row, output in tqdm(zip(test_rows, outputs), total=len(test_rows)):
        objects_wikiid = []
        objects = []
        scores=[]
        for seq in output:
            obj = seq["token_str"]
            score = seq["score"]
            # if obj in negative_vocabulary:
            #     continue

            if obj == config.EMPTY_TOKEN:
                # objects_wikiid.append(config.EMPTY_STR)
                obj = ''
                # objects = [config.EMPTY_STR]
                # break
            # else:
                # Perform disambiguation using a baseline method for the object
            # wikidata_id = util.disambiguation_baseline(obj)
            wikidata_id=0
            objects_wikiid.append(wikidata_id)

            objects.append(obj)
            scores.append(score)

        # if config.EMPTY_STR in objects:
        #     objects = [config.EMPTY_STR]

        # Create a result row with the subject entity, object entities, and relation
        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": objects_wikiid,
            "ObjectEntities": objects,
            "Relation": row["Relation"],
            "ObjectEntitiesScore":scores
        }
        results.append(result_row)

    # Save the results to the specified output file
    logger.info(f"Saving the results to \"{args.output_fn}\"...")
    util.file_write_json_line(args.output_fn, results)

    logger.info(f"Start Evaluate ...")
    evaluate.evaluate(args.output_fn, args.test_fn)

    evaluate.assign_label(args.output_fn, args.test_fn)
  

def test_pipeline_origin():
    # Load the configuration for the trained BERT model from the specified directory
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)

    # Load the trained BERT model for masked language modeling
    bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_best_dir, config=bert_config
    )

    # Create a pipeline for the specified task using the loaded BERT model and tokenizer
    pipe = pipeline(
        task=task,
        model=bert_model,
        tokenizer=bert_tokenizer,
        top_k=args.top_k,
        device=args.gpu,
    )

    # Read the prompt templates from the specified file
    prompt_templates = util.file_read_prompt(args.template_fn)

    # Read the test data rows from the specified file
    test_rows = util.file_read_json_line(args.test_fn)

    prompts = []
    for row in test_rows:
        # Generate a prompt for each test row using the corresponding template
        prompt = prompt_templates[row["Relation"]].format(
            subject_entity=row["SubjectEntity"],
            mask_token=bert_tokenizer.mask_token,
        )
        prompts.append(prompt)

    logger.info(f"Running the model...")

    # Run the model on the generated prompts in batches
    outputs = pipe(prompts, batch_size=args.train_batch_size * 4)

    logger.info(f"End the model...")

    results = []
    rel_thres_fn = f"{config.RES_DIR}/relation-threshold.json"

    # use the saved threshold of each relationship
    # a adaptive threshold or tok-k function is are the works
    if os.path.exists(rel_thres_fn):
        with open(rel_thres_fn, 'r') as f:
            rel_thres_dict = json.load(f)
    else:
        rel_thres_dict = dict()

    for row, output, prompt in zip(test_rows, outputs, prompts):
        objects_wikiid = []
        objects = []

        for seq in output:
            # Check if the relation is present in the relation-threshold dictionary
            if row[config.KEY_REL] not in rel_thres_dict:
                print(f"{row[config.KEY_REL]} not in rel_thres_dict")

                # Assign a threshold value for the relation if it's not present
                rel_thres_dict[row[config.KEY_REL]] = args.threshold

            # Filter the output sequence based on the relation threshold
            if seq["score"] > rel_thres_dict[row[config.KEY_REL]]:
                obj = seq["token_str"]

                if obj == config.EMPTY_TOKEN:
                    objects_wikiid.append(config.EMPTY_STR)
                    objects = [config.EMPTY_STR]
                    break
                else:
                    # Perform disambiguation using a baseline method for the object
                    wikidata_id = util.disambiguation_baseline(obj)
                    objects_wikiid.append(wikidata_id)

                objects.append(obj)

        if config.EMPTY_STR in objects:
            objects = [config.EMPTY_STR]

        # Create a result row with the subject entity, object entities, and relation
        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": objects_wikiid,
            "ObjectEntities": objects,
            "Relation": row["Relation"],
        }
        results.append(result_row)

    # Save the results to the specified output file
    logger.info(f"Saving the results to \"{args.output_fn}\"...")
    util.file_write_json_line(args.output_fn, results)

    logger.info(f"Start Evaluate ...")
    evaluate.evaluate(args.output_fn, args.test_fn)

    evaluate.assign_label(args.output_fn, args.test_fn)


# def topk_process(input_list):
#     relation, pred_list = input_list
#     predefine_fine = 'res/object_number.tsv'
#     with open(predefine_fine) as f:
#         topk = csv.DictReader(f, delimiter="\t")
#         topk_max_dict = { row["Relation"]:eval(row['Val'])[1] for row in topk}
#     groud_list = relation_list_groud[relation]
#     origin_topk = topk_max_dict[relation]
#     best_f1=0
#     best_topk=0
#     for i in tqdm(range(origin_topk*2,origin_topk//2,-1)):
#         for row in pred_list:
#             row['ObjectEntities'] = row['ObjectEntities'][:i]
#         f1 = evaluate.evaluate_list(groud_list, pred_list)[relation]['f1']
#         if f1> best_f1:
#             best_f1=f1
#             best_topk =i

#     return best_topk,best_f1


def adaptive_top_k():
    origin_result_dict = evaluate.evaluate(args.output_fn, args.test_fn)
    predefine_fine = 'res/object_number.tsv'
    with open(predefine_fine) as f:
        topk = csv.DictReader(f, delimiter="\t")
        topk_max_dict = { row["Relation"]:eval(row['Val'])[1] for row in topk}
    pred_rows = util.file_read_json_line(args.output_fn)
    groud_rows = util.file_read_json_line(args.test_fn)
    relation_list_pred = dict()
    relation_list_groud = dict()
    best_topk_dict=dict()
    for row in pred_rows:
        relation = row['Relation']
        if relation not in relation_list_pred:
            relation_list_pred[relation]=[]
        relation_list_pred[relation].append(row)

    for row in groud_rows:
        relation = row['Relation']
        if relation not in relation_list_groud:
            relation_list_groud[relation]=[]
        relation_list_groud[relation].append(row)

    for relation, pred_list in tqdm(relation_list_pred.items()):
        groud_list = relation_list_groud[relation]
        origin_topk = topk_max_dict[relation]
        best_f1=0
        best_topk=0
        for i in range(origin_topk*3, 0,-1):
            if i >= len( pred_list[0]['ObjectEntities']):
                continue
            for row in pred_list:
                row['ObjectEntities'] = row['ObjectEntities'][:i]
            f1 = evaluate.evaluate_list(groud_list, pred_list)[relation]['f1']
            if f1> best_f1:
                best_f1=f1
                best_topk =i
        best_topk_dict[relation] = [best_topk,best_f1] 
        origin_result_dict[relation]["best_topk"]=best_topk
        origin_result_dict[relation]["best_f1"]=best_f1
        origin_result_dict[relation]["origin_topk"]=origin_topk
    

    origin_result_dict["Average"]["best_f1"] =  sum([x["best_f1"] if "best_f1" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    result_dict = {
        "args":args.__dict__,
        "metric":origin_result_dict
        }
    util.file_write_json_line(config.RESULT_FN, [result_dict],'auto')
    scores_per_relation_pd = pd.DataFrame(origin_result_dict)
    print(scores_per_relation_pd.transpose().round(2))



def adaptive_threshold():
    origin_result_dict = evaluate.evaluate(args.output_fn, args.test_fn)
    predefine_fine = 'res/object_number.tsv'
    with open(predefine_fine) as f:
        topk = csv.DictReader(f, delimiter="\t")
        topk_max_dict = { row["Relation"]:eval(row['Val'])[1] for row in topk}
    threshold_initial_dict = dict()
    for k,v in topk_max_dict.items():
        threshold_initial_dict[k] = 1/float(v)
    pred_rows = util.file_read_json_line(args.output_fn)
    groud_rows = util.file_read_json_line(args.test_fn)
    relation_list_pred = dict()
    relation_list_groud = dict()
    best_topk_dict=dict()
    for row in pred_rows:
        relation = row['Relation']
        if relation not in relation_list_pred:
            relation_list_pred[relation]=[]
        relation_list_pred[relation].append(row)

    for row in groud_rows:
        relation = row['Relation']
        if relation not in relation_list_groud:
            relation_list_groud[relation]=[]
        relation_list_groud[relation].append(row)

    for relation, pred_list in tqdm(relation_list_pred.items()):
        groud_list = relation_list_groud[relation]
        origin_topk = topk_max_dict[relation]
        best_f1=0
        origin_threshold = threshold_initial_dict[relation]
        best_threshold=0
        for i in range(1,int((origin_threshold*3)/0.05),1):
            threshold = 0.05*i
            for row in pred_list:
                row['ObjectEntities'] =list(map(lambda t:t[1], filter(lambda t: t[0]>=threshold, zip(row['ObjectEntitiesScore'],  row['ObjectEntities']))))
            f1 = evaluate.evaluate_list(groud_list, pred_list)[relation]['f1']
            if f1> best_f1:
                best_f1=f1
                best_threshold =threshold
        best_topk_dict[relation] = [best_threshold,best_f1] 
        origin_result_dict[relation]["best_threshold"]=best_threshold
        origin_result_dict[relation]["best_f1"]=best_f1
        origin_result_dict[relation]["best_threshold"]=origin_threshold
    

    origin_result_dict["Average"]["best_f1"] =  sum([x["best_f1"] if "best_f1" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    result_dict = {
        "args":args.__dict__,
        "metric":origin_result_dict
        }
    util.file_write_json_line(config.RESULT_FN, [result_dict],'auto')
    scores_per_relation_pd = pd.DataFrame(origin_result_dict)
    print(scores_per_relation_pd.transpose().round(2))



    # for relation, v in best_topk_dict.items():
    #     print(relation,v[1],v[0], topk_max_dict[relation] )
    # # print(json.dumps(best_topk_dict, indent=4))
    # average_f1 = sum(map(lambda x:x[1], best_topk_dict.values()))/ len(best_topk_dict)
    # print("average_f1",average_f1)



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


    parser.add_argument(
        "-i", "--test_fn", type=str, required=True, help="Input test file (required)"
    )
    parser.add_argument(
        "-o", "--output_fn", type=str, required=True, help="Output file (required)"
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
    negative_vocabulary_fn = f'data/negave_vocabulary.json'
    #with open(negative_vocabulary_fn) as f:
    #    negative_vocabulary = set(json.load(f))
    if "train" in args.mode:
        train()

    if "test" in args.mode:
        test_pipeline()
        adaptive_threshold()
