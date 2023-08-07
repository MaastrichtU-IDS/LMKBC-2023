import csv
import itertools
import json
import argparse
import logging
import random
import pandas as pd
#from tokenizers import Tokenizer

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import transformers
from transformers import BertTokenizer, pipeline, BertTokenizerFast, BertModel, BertPreTrainedModel,BertForMaskedLM
import os
from glob import glob

import config
#os.environ['TRANSFORMERS_CACHE'] = config.TRANSFOER_CACHE_DIR
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

print("GPU ", torch.cuda.is_available())

with open(config.TOKENIZER_PATH+"/added_tokens.json") as f:
    additional_token_dict = json.load(f)

type_entity_fp =f'res/entity_for_tokenizer.json'
type_entity_dict = json.load(open(type_entity_fp))
for k,v in type_entity_dict.items():
    type_entity_dict[k]=set(v)
print('type_entity_dict',len(type_entity_dict))
# print(type_entity_dict)
class MLMDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizerFast, data_fn, template_fn,silver_data=False) -> None:
        super().__init__()
        self.data = []

        # Read the training data from a JSON file
        train_data = util.file_read_json_line(data_fn)
        if silver_data:
            print("silver data")
            file_name_list = glob(f'res/silver/*.jsonl', recursive=True)
            for fp in file_name_list:
                train_data.extend(util.file_read_json_line(fp)) 

        # Read the prompt templates from a file
        prompt_templates = util.file_read_prompt(template_fn)

        # Iterate over each row in the training data
        for row in tqdm(train_data):
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
        print("data set size",len(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class MLM_Multi_Dataset(Dataset):
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
                obj_tokens = tokenizer.tokenize(obj)
                if len(obj_tokens)> config.mask_length:
                    obj_tokens =obj_tokens[:config.mask_length]
                elif len(obj_tokens)< config.mask_length:
                    obj_tokens=obj_tokens+[tokenizer.pad_token] * (config.mask_length-len(obj_tokens))
                # Create an input sentence by formatting the prompt template
                input_sentence = prompt_template.format(
                    subject_entity=subject, mask_token=' '.join([tokenizer.mask_token] * config.mask_length)
                )

                # Convert the object entity to its corresponding ID
                obj_id = tokenizer.convert_tokens_to_ids(obj_tokens)

                input_ids = tokenizer.encode(input_sentence)
                attention_mask = [0 if v == tokenizer.mask_token_id else 1 for v in input_ids]
                mask_index= [ i for i,v in enumerate(input_ids) if v == tokenizer.mask_token_id]
                # Create label IDs where the masked token corresponds to the object ID
                label_ids = [
                    obj_id[i-mask_index[0]] if t == tokenizer.mask_token_id else -100 for i, t in enumerate(input_ids)
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
    if not os.path.isdir( args.model_load_dir) and args.token_recode:
        print("repair token embedding")
        util.token_layer(bert_model, additional_token_dict,enhance_tokenizer, origin_tokenizer)
        # bert_model.resize_token_embeddings(len(toke))
    # else:
    #     print(f"using huggingface  model {args.model_load_dir}")
    #     bert_config = transformers.AutoConfig.from_pretrained(config.bert_base_cased)
    #     bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
    #         config.bert_base_cased, config=bert_config
    #     )

    # is_train can be used to resume train process in formal training
    is_train = os.path.isdir(args.model_load_dir)
    tokenizer = enhance_tokenizer
    bert_model.resize_token_embeddings(len(tokenizer))
    # Create the training dataset using the MLMDataset class, providing the train data file, tokenizer, and template file
    train_dataset = MLMDataset(
        data_fn=args.train_fn, tokenizer=tokenizer, template_fn=args.template_fn,
        silver_data=args.silver_data,
    )
    print("trainset size",len(train_dataset))
    # transformers.BertForMaskedLM()

    # Create a data collator for batching and padding the data during training
    bert_collator = util.DataCollatorKBC(
        tokenizer=tokenizer,
    )

    # Create the development dataset using the MLMDataset class, providing the dev data file, tokenizer, and template file
    dev_dataset = MLMDataset(
        data_fn=args.dev_fn,
        tokenizer=tokenizer,
        template_fn=args.template_fn,
    )

    # Resize the token embeddings of the BERT model to match the tokenizer's vocabulary size
    # bert_model.resize_token_embeddings(len(bert_tokenizer))

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
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=args.model_best_dir)
    print(f"model_best_dir: ", args.model_best_dir)



# def format_output(model_output,tokenizer):
#     logits = result.logits
#     logits = torch.softmax(logits,-1)
#     print(logits.shape)
#     # to("cpu")
#     logits=logits[0,mask_indexs[0]:mask_indexs[-1]]
#     logits_topk = torch.topk(logits, 20,-1)
#     values = logits_topk[0].to('cpu').detach().numpy()
#     indices = logits_topk[1].to('cpu').detach().numpy() 
#     print("indices ",indices)
#     print("values ",values)
#     for i,v in zip(indices,values):
#         for ei,ev in zip(i,v):
#             print(bert_tokenizer.convert_ids_to_tokens(int(ei)), ei,ev)

#     # logits = logits_topk[mask_indexs[0]:mask_indexs[-1]]

#     print(logits_topk)
#     # input_ids.append(input_id)
#     # prompts.append(prompt)


def search_entity(results):
    probability = [] 
    for r in results:
        # print(r)
        token_list = list(filter(lambda x:len(x) > 0, map(lambda x:x[0], r)))
        # print(token_list)
        comb = list(itertools.product(*token_list))
        probability.append(comb)
        # probability.append()
    return probability

def predict():
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)
    bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_best_dir, config=bert_config
    )
    tokenizer =tokenizer
    token_mapping_fp = config.RES_DIR+"/token_mapping.json"
    with open(token_mapping_fp) as f:
        entity_maaping = json.load(f)
    # Resize the token embeddings of the BERT model to match the tokenizer's vocabulary size
    # bert_model.resize_token_embeddings(len(bert_tokenizer))
    # bert_model.forward()
    # Set up the training arguments for the model
    
        # Read the prompt templates from the specified file
    prompt_templates = util.file_read_prompt(args.template_fn)

    # Read the test data rows from the specified file
    test_rows = util.file_read_json_line(args.test_fn)

    # bert_model.training=False
    # bert_model = bert_model.to('cuda:0')
    results = []
    exclusive_list = {config.EMPTY_TOKEN,tokenizer.pad_token,'(',")"}
    exclusive_ids =set( tokenizer.convert_tokens_to_ids(exclusive_list))

    
    for row in test_rows:
        # Generate a prompt for each test row using the corresponding template
        prompt = prompt_templates[row["Relation"]].format(
            subject_entity=row["SubjectEntity"],
            mask_token=' '.join([tokenizer.mask_token] *config.mask_length)
        )
    
        tokens = tokenizer.tokenize(prompt)
        # print(tokens)
        mask_indexs = [i for i, x in enumerate(tokens) if x == tokenizer.mask_token]
        # print(mask_indexs)
        input_id = tokenizer.encode(prompt,  return_tensors='pt')
        # print(tokenizer.convert_ids_to_tokens(input_id))
        # input_id=torch.tensor(input_id).unsqueeze(0)
        # print("input_id",input_id.shape)
        result = bert_model.forward(input_id)
        logits = result.logits
        logits = torch.softmax(logits,-1)
        # print(logits}.shape)
        # to("cpu")
        logits=logits[0,mask_indexs[0]:mask_indexs[-1]]
        logits_topk = torch.topk(logits, 20,-1)
        values = logits_topk[0].detach().numpy()
        indices = logits_topk[1].detach().numpy()
        mask_token_list= []
        for i,v in zip(indices,values):
            token_list=[]
            score_list=[]
            for ei,ev in zip(i,v):
                if ev > 0.05:
                    token_list.append(tokenizer.convert_ids_to_tokens(int(ei)))
                    score_list.append(float(ev))
                # print(bert_tokenizer.convert_ids_to_tokens(int(ei)), ev)
            mask_token_list.append((token_list,score_list))
        # logits = logits_topk[mask_indexs[0]:mask_indexs[-1]]
        results.append(mask_token_list)
        # print(results[0])
    print("results" , results[:10])
    re = search_entity(results)
    result_list = [] 
    print("re" , re[:10])
    for r,row in zip(re,test_rows):
        # print("r ", r)
        objects = []
        objects_wikiid=[]

        for rs in r:
            if rs[0] ==  "Empty":
                 rs_t=["Empty"]
            else:
                rs_t = []
                for t in rs:
                    if t not in exclusive_list:
                        rs_t.append(t)
                    else:
                        break
            # print(rs)
            # if len(rs) == 1: 
            #     objects.append(rs[0])
            #     continue
            
            key_rs = ' '.join(rs_t)
            if key_rs not in entity_maaping:
                continue
            entity =  entity_maaping[key_rs][0]
            objects.append(entity)

        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": objects_wikiid,
            "ObjectEntities": objects,
            "Relation": row["Relation"],
        }
        result_list.append(result_row)
    # print("result_list ",result_list[:30])
    print(json.dumps(result_list[:10],indent = 2))
    util.file_write_json_line(args.output_fn, result_list)
    logger.info(f"Start Evaluate ...")
    evaluate.evaluate(args.output_fn, args.test_fn)


def test_pipeline():
    # Load the configuration for the trained BERT model from the specified directory
    bert_config = transformers.AutoConfig.from_pretrained(args.model_best_dir)

    # Load the trained BERT model for masked language modeling
    bert_model: BertModel = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_best_dir, config=bert_config,ignore_mismatched_sizes=True
    )

    # Create a pipeline for the specified task using the loaded BERT model and tokenizer
    pipe = pipeline(
        task=task,
        model=bert_model,
        tokenizer=enhance_tokenizer,
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
            mask_token=enhance_tokenizer.mask_token,
        )
        prompts.append(prompt)

    logger.info(f"Running the model...")

    # Run the model on the generated prompts in batches
    outputs = pipe(prompts, batch_size=args.train_batch_size * 4)

    logger.info(f"End the model...")

    results = []
    exclusive_token= {"the","of"}
    results = []

    if args.filter:
        print('filter"')
    for row, output in zip(test_rows, outputs):
        objects_wikiid = []
        objects = []
        scores=[]
        relation = row["Relation"]
        entity_type = config.relation_entity_type_dict[relation][1]
        type_entity_set = type_entity_dict[entity_type]
        # print(entity_type,len(type_entity_set))
        # print("type_entity_set",type_entity_set)
        for seq in output:
            obj = seq["token_str"]
            score = seq["score"]
            # if obj in negative_vocabulary:
            #     continue
            if args.filter:
                # print('entity filter')
                # print('type_entity',len(type_entity))
                if obj not in type_entity_set:
                    # print(obj)
                    continue
            if obj.startswith("##"):
                continue
            if obj == config.EMPTY_TOKEN:
                obj = ''
            
            wikidata_id= util.disambiguation_baseline(obj)
            objects_wikiid.append(wikidata_id)

            objects.append(obj)
            scores.append(score)
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
        tokenizer=enhance_tokenizer,
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
            mask_token=enhance_tokenizer.mask_token,
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
        best_precision = 0
        best_recal= 0 
        origin_threshold = threshold_initial_dict[relation]
        best_threshold=0
        threshold_step = 0.01
        for i in range(1,int((origin_threshold*3)/threshold_step),1):
            threshold = threshold_step*i
            for row in pred_list:
                row['ObjectEntities'] =list(map(lambda t:t[1], filter(lambda t: t[0]>=threshold, zip(row['ObjectEntitiesScore'],  row['ObjectEntities']))))
            eval_dict = evaluate.evaluate_list(groud_list, pred_list)[relation]
            f1 = eval_dict['f1']
            p = eval_dict['p']
            r = eval_dict['r']
            if f1> best_f1:
                best_f1=f1
                best_threshold =threshold
                best_precision= p 
                best_recal= r
        best_topk_dict[relation] = [best_threshold,best_f1] 
        origin_result_dict[relation]["best_threshold"]=best_threshold
        origin_result_dict[relation]["best_f1"]=best_f1
        origin_result_dict[relation]["best_precision"]=best_precision
        origin_result_dict[relation]["best_recal"]=best_recal
        #origin_result_dict[relation]["origin_threshold"]=origin_threshold
    

    origin_result_dict["Average"]["best_f1"] =  sum([x["best_f1"] if "best_f1" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    origin_result_dict["Average"]["best_precision"] =  sum([x["best_precision"] if "best_precision" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    origin_result_dict["Average"]["best_recal"] =  sum([x["best_recal"] if "best_recal" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    result_dict = {
        "args":args.__dict__,
        "metric":origin_result_dict
        }
    util.file_write_json_line(config.RESULT_FN, [result_dict],'auto')
    scores_per_relation_pd = pd.DataFrame(origin_result_dict)
    print(scores_per_relation_pd.transpose().round(2).to_string(max_cols=12))



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
        "--token_recode",
        type=bool,
        default=False,
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
        type=float,
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
        "--pretrain_model",
        type=str,
        required=False,
        default=config.bert_base_cased,
        help="CSV file containing train data for few-shot examples (required)",
    )
        
    parser.add_argument(
        "--add_corpus",
        type=bool,
        required=False,
        default=False,
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

    
    parser.add_argument(
        "--silver_data",
        type=bool,
        default=False,
        help="Batch size for the model. (default:32)",
    )

    parser.add_argument(
        "--filter",
        type=bool,
        default=False,
        help="Batch size for the model. (default:32)",
    )

    parser.add_argument(
        "--label",
        type=str,
        default='null',
        help="Batch size for the model. (default:32)",
    )


    args = parser.parse_args()
    tokenizer_dir = f'{config.RES_DIR}/tokenizer/bert'
    enhance_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
    origin_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_load_dir)
    negative_vocabulary_fn = f'data/negave_vocabulary.json'
    #with open(negative_vocabulary_fn) as f:
    #    negative_vocabulary = set(json.load(f))
    if "train" in args.mode:
        train()

    if "test" in args.mode:
        test_pipeline()
        adaptive_threshold()

    if "predict" in args.mode:
        predict()
        # adaptive_threshold()
