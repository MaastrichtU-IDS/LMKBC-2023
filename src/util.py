import csv
import json
import os
import random
from typing import List

import requests
import torch
from transformers import BertTokenizerFast,BertModel
import transformers
print("getcwd", os.getcwd())

import config

from typing import List, Union, Dict, Any, Optional, Mapping
import numpy as np

local_cache_path = f'{config.RES_DIR}/item_cache.json'
local_cache = dict()
if os.path.exists(local_cache_path):
    local_cache = json.load(open(local_cache_path))


# Disambiguation baseline
def disambiguation_baseline(entity_label):
    if entity_label in local_cache:
        return local_cache[entity_label]
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entity_label}&language=en&format=json"
        data = requests.get(url).json()
        # Return the first id (Could upgrade this in the future)
        first_id = data['search'][0]['id']
        local_cache[entity_label] = first_id
        return first_id
    except:
        return entity_label
    
def save_entity_id():
    with open(local_cache_path, "w") as f:
        json.dump(local_cache, f, indent = 2)



# Read prompt templates from a CSV file
def file_read_prompt(file_path: str):
    # print('file_path', file_path)
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
    return prompt_templates


def line_to_json(line: str):
    train_data = []
    if len(line) == 0:
        return train_data
    try:
        if line.find('}{') != -1:
            line = line.replace('}{', '}\n{')
            line_list = line.split('\n')
            for l in line_list:
                train_data.extend(line_to_json(l))
        else:
            train_data.append(json.loads(line))
    except Exception as e:
        print(line)
        raise e
    return train_data


def file_read_json_line(data_fn):
    if os.path.exists(data_fn) and os.path.isfile(data_fn):
        with open(data_fn, "r") as file:
            lines = file.readlines()
            train_data = []
            for line in lines:
                train_data.extend(line_to_json(line))
    train_data = []
    try:
        with open(data_fn, "r") as file:
            lines = file.readlines()
            for line in lines:
                train_data.append(json.loads (line))
    except :
        print(line)

    return train_data


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def flat_list(data_list: list):
    datas = []
    for data in data_list:
        if isinstance(data, list):
            datas.extend(flat_list(data))
        else:
            datas.append(data)
    return datas


def file_write_json_line(data_fn, results, mode='w'):
    results = flat_list(results)
    json_text_list = [json.dumps(aj, cls=SetEncoder) for aj in results]
    file_write_line(data_fn, json_text_list, mode)

def file_delete(fp):
    if os.path.exists(fp):
        os.remove(fp)

def file_write_line(data_fn, results, mode='w'):
    if mode == 'auto':
        mode = 'a' if os.path.exists(data_fn) else 'w'
    with open(data_fn, mode) as f:
        text = '\n'.join(results)
        if mode == 'a':
            text = '\n'+text
        f.write(text)


def create_prompt(
    subject_entity: str,
    relation: str,
    prompt_templates: dict,
    instantiated_templates: List[str],
    tokenizer,
    few_shot: int = 0,
    task: str = "fill-mask",
) -> str:
    prompt_template = prompt_templates[relation]
    if task == "text-generation":
        if few_shot > 0:
            random_examples = random.sample(
                instantiated_templates, min(few_shot, len(instantiated_templates))
            )
        else:
            random_examples = []
        few_shot_examples = "\n".join(random_examples)
        prompt = f"{few_shot_examples}\n{prompt_template.format(subject_entity=subject_entity)}"
    else:
        prompt = prompt_template.format(
            subject_entity=subject_entity, mask_token=tokenizer.mask_token
        )
    return prompt


def recover_mask_word_func(mask_word, bert_tokenizer):
    word_resume = bert_tokenizer.convert_tokens_to_string(mask_word)
    index_padding = word_resume.find(bert_tokenizer.padding_token)
    if index_padding > -1:
        word_resume = word_resume[:index_padding]
    return word_resume


class KnowledgeGraph:
    def __init__(self, data_fn, kg=None):
        # read train file, each line is a josn object
        train_line = file_read_json_line(data_fn)
        # if parameter kg is none, create a new dict object. 
        # else use the parameter kg. 
        self.kg = dict() if kg is None else kg
        for row in train_line:
            relation = row['Relation']
            object_entities = row['ObjectEntities']
            subject = row["SubjectEntity"]
            self.add_triple(subject, relation, object_entities)

    def ensure_key_exists_for_entity(self, entity, relation):
        # make sure the basic key for each entity exists
        if entity not in self.kg:
            self.kg[entity] = dict()
        if config.TO_KG not in self.kg[entity]:
            self.kg[entity][config.TO_KG] = dict()
        if config.FROM_KG not in self.kg[entity]:
            self.kg[entity][config.FROM_KG] = dict()

        if relation not in self.kg[entity][config.TO_KG]:
            self.kg[entity][config.TO_KG][relation] = set()
        if relation not in self.kg[entity][config.FROM_KG]:
            self.kg[entity][config.FROM_KG][relation] = set()

    def add_triple(self, subject, relation, object_entities):
        # for example: 
        # {
        #     "The Netherlands": {
        #         "to": {
        #             "CountryBordersCountry": [
        #                 "Germany"
        #             ]
        #         },
        #         "from": {
        #             "CountryBordersCountry": [
        #                 "Belgium"
        #             ]
        #         }
        #     }
        # }
        if subject == config.EMPTY_STR:
            return 
        if not isinstance(object_entities, (list, set)):
            object_entities = [object_entities]

        self.ensure_key_exists_for_entity(subject, relation)
        self.kg[subject][config.TO_KG][relation].update(object_entities)
        for entity in object_entities:
            if entity == '':
                continue 
            self.ensure_key_exists_for_entity(entity, relation)
            self.kg[entity][config.FROM_KG][relation].add(subject)

    def __getitem__(self, index):
        if index in self.kg:
            return self.kg[index]
        else:
            return None

    def __contains__(self, item):
        return item in self.kg


class DataCollatorKBC:
    def __init__(self, tokenizer: BertTokenizerFast, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # examples = examples.clone()
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_attention_mask=True,
        )
        if self.padding:
            max_length = len(batch['input_ids'][0])
            label_list = []
            for label in batch['labels']:
                length_diff = max_length - len(label)
                label1 = label.copy()
                if length_diff > 0:
                    pad_list = [-100] * (length_diff)
                    label1.extend(pad_list)

                label_list.append(label1)
            batch['labels'] = label_list

        batch_pt = dict()
        for k, v in batch.items():
            batch_pt[k] = torch.tensor(v)

        return batch_pt


def tokenize_sentence(tokenizer, input_sentence: str):
    input_tokens = (
        [tokenizer.cls_token]
        + tokenizer.tokenize(input_sentence)
        + [tokenizer.sep_token]
    )
    # input_tokens = tokenizer.tokenize(input_sentence)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [0 if v == tokenizer.mask_token else 1 for v in input_tokens]

    return input_ids, attention_mask


def softmax(x, axis=0):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# if __name__ == "__main__":
#     list_1 = [[1], [1, 2], [[1], [2], [2, 3, [4, 5, [7]]]]]
#     list_flat = flat_list(list_1)
#     print(list_flat)


class Printer:
    def __init__(self, times):
        self.times=times
        self.channel_times=dict()

    def __call__(self, obj, channel='default'):
        if channel not in self.channel_times:
            self.channel_times[channel]=self.times
        if self.channel_times[channel] > 0:
            print(obj)
            self.channel_times[channel]-=1

def assemble_result(origin_rows, outputs):
    results = []
    for row, output in zip(origin_rows, outputs):
        objects_wikiid = []
        objects = []
        scores=[]
        for seq in output:
            obj = seq["token_str"]
            score = seq["score"]
            # if obj in negative_vocabulary:
            #     continue
            if obj.startswith("##"):
                continue
            if obj == config.EMPTY_TOKEN:
                obj = ''
            wikidata_id= disambiguation_baseline(obj)
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
    return results 



def token_layer(model:transformers.BertForMaskedLM, enhance_tokenizer, origin_tokenizer:transformers.BertweetTokenizer,recode_type):
    # BertForMaskedLM.get_input_embeddings()
    # BertForMaskedLM.set_input_embeddings()
    with open(config.TOKENIZER_PATH+"/added_tokens.json") as f:
        additional_token_dict = json.load(f)

    num_new_tokens = len(enhance_tokenizer.vocab)
    # model.resize_token_embeddings(num_new_tokens)

    old_token_embedding = model.get_input_embeddings()
    
    old_num_tokens, old_embedding_dim = old_token_embedding.weight.shape
    old_output_embedding =  model.get_output_embeddings()
    # print("old_output_embedding  ", old_output_embedding.weight.data.dtype)
    # print("old_output_embedding  ", old_output_embedding.weight.data.shape)
    old_output_dim_0, old_output_dim_1 =   old_output_embedding.weight.shape
    # print()
    # new_embeddings = torch.nn.Module()
    new_input_embeddings = torch.nn.Embedding(
         num_new_tokens, old_embedding_dim
    )
    cls_bias = torch.zeros(num_new_tokens)
    new_cls_decoder = torch.nn.Linear(
         old_output_dim_1, num_new_tokens, dtype= model.cls.predictions.decoder.weight.dtype
    )
    new_output_embeddings = torch.nn.Linear(
         old_output_dim_1, num_new_tokens, dtype= old_output_embedding.weight.dtype
    )
    # print("new_output_embeddings  ", new_output_embeddings.weight.data.dtype)
    # print("new_output_embeddings  ", new_output_embeddings.weight.data.shape)
    # print("new_cls_decoder  ", new_cls_decoder.weight.data.shape)
    # embedding_laye_dictr =embedding_layer . state_dict()
    # print("embedding_laye_dictr",embedding_laye_dictr.keys())
    # embedding_laye_dictr['weight']=new_embeddings.state_dict()['weight']
    # new_embeddings.load_state_dict(embedding_laye_dictr)

    # Creating new embedding layer with more entries
    # new_embeddings.weight = torch.nn.Parameter(old_num_tokens+num_new_tokens, old_embedding_dim)
 
    # Setting device and type accordingly
    new_output_embeddings = new_output_embeddings.to(
        old_output_embedding.weight.device,
        dtype=old_output_embedding.weight.dtype,
    )
    new_input_embeddings = new_input_embeddings.to(
        old_token_embedding.weight.device,
        dtype=old_token_embedding.weight.dtype,
    )

    # Copying the old entries
    new_input_embeddings.weight.data[:old_num_tokens, :] = old_token_embedding.weight.data[
        :old_num_tokens, :    ]
    new_output_embeddings.weight.data[:old_num_tokens, :] = old_output_embedding.weight.data[
        :old_num_tokens, :    ]
    cls_bias[ :old_num_tokens] = model.cls.predictions.bias.data[ :old_num_tokens] 
    new_cls_decoder.weight.data[:old_num_tokens, :] = model.cls.predictions.decoder.weight.data[
        :old_num_tokens, :    ]
    # old_position = model.get_position_embeddings()
    # position_dim_0, position_dim_1 = old_position.weight.shape
    # position_dim_0_new = position_dim_0+len(additional_token_dict)
    # new_position = torch.nn.Embedding(
    #      position_dim_0_new, old_embedding_dim
    # )
    # new_embeddings.padding_idx = embedding_layer.clone()
    print_count= 0
    if recode_type == 'std':
        tw = Token_Weight(origin_tokenizer)
    for entity,index in additional_token_dict.items():
        token_ids = origin_tokenizer.encode(entity,add_special_tokens=False)
      
        old_output = old_output_embedding.weight.data[token_ids,:]
        old_input = old_token_embedding.weight.data[token_ids,:]
        old_cls_bias = model.cls.predictions.bias.data[token_ids]
        old_cls_decoder_data = model.cls.predictions.decoder.weight.data[token_ids,:]

        # print("old_output",old_output.shape)
        # print("weight",weight.shape)
        if recode_type == 'std':
            weight = tw.token_weight(token_ids)
            new_output = torch.multiply(old_output,weight).sum(dim=0)
            # new_output = torch.nn.functional.normalize(new_output, p=2, dim=0)
            if len (token_ids) == 2 and print_count < 10:
            # and Fals
                print_count+=1
                print('entity',entity)
                print('tokens',origin_tokenizer.convert_ids_to_tokens(token_ids))
                print('old_output',old_output[:,:5])
                print('weight',weight)
                print('new_output',new_output[:5])
                print()
            new_input =  torch.multiply(old_input,weight).sum(dim=0)
            # new_input = torch.nn.functional.normalize(new_input, p=2, dim=0)
            # print('old_cls_bias',old_cls_bias.shape)
            new_cls_bias = torch.multiply(old_cls_bias,weight).sum()
            # print('new_cls_bias',new_cls_bias.shape)
            new_cls_decoder_data =torch.multiply(old_cls_decoder_data,weight).sum(dim=0)
            # new_cls_decoder_data = torch.nn.functional.normalize(new_cls_decoder_data, p=2, dim=0)
            
        elif recode_type == 'mean':
            new_output =   torch.mean(old_output,0,keepdim = True)
            new_input =  torch.mean(old_input,0,keepdim = True)
            new_cls_bias = torch.mean(old_cls_bias)
            new_cls_decoder_data = torch.mean(old_cls_decoder_data,0,keepdim = True)
        elif recode_type == 'min':
            new_output =   torch.min(old_output,0,keepdim = True)[0]
            new_input =  torch.min(old_input,0,keepdim = True)[0]
            new_cls_bias = torch.min(old_cls_bias)
            new_cls_decoder_data = torch.min(old_cls_decoder_data,0,keepdim = True)[0]
        elif recode_type == 'max':
            new_output =   torch.max(old_output,0,keepdim = True)[0]
            new_input =  torch.max(old_input,0,keepdim = True)[0]
            new_cls_bias = torch.max(old_cls_bias)
            new_cls_decoder_data = torch.max(old_cls_decoder_data,0,keepdim = True)[0]
        else:
            raise Exception('recode_type should be in (max, min, std, mean)')


        new_output_embeddings.weight.data[index, :] =new_output
        new_input_embeddings.weight.data[index,:] =new_input
        cls_bias[index] = new_cls_bias
        new_cls_decoder.weight.data[index,:] = new_cls_decoder_data
       
        # new_position .weight.data[index,:] = new_position_token
    # print("old_token_embedding ", old_token_embedding.weight.data[2][:5])
    # print("new_token_embeddings ", new_input_embeddings.weight.data[2][:5])
    # print("new_output_embeddings ", new_output_embeddings.weight.data[2][:5])
    # print("old_output_embedding ", old_output_embedding.weight.data[2][:5])
    model.set_input_embeddings( new_input_embeddings)
    model.set_output_embeddings( new_output_embeddings)
    model.cls.predictions.bias.data =  cls_bias
    model.cls.predictions.decoder =  new_cls_decoder
    model.config.vocab_size = num_new_tokens
    model.vocab_size = num_new_tokens
    return model

class Token_Weight:
    def __init__(self,origin_tokenizer):
        self._weight:dict = json.load(open(config.token_count_fp))
        self.origin_tokenizer = origin_tokenizer

    def token_weight(self, token_ids:list):
        counts = [self._weight[str(tid)] if str(tid) in self._weight else 10000 for tid in token_ids]
        count_tensor = torch.tensor(counts,dtype=torch.float32)

        # weights = count_tensor/torch.sum(count_tensor)
        # min_value, min_index = torch.min(weights,0)
        # # max_value, max_index = torch.max(weights,0)
        # weight = -1*weights
        # weight[min_index] = 1
        # return weight.unsqueeze(1)

        weights = 1/count_tensor
        weights = weights.pow(1/4)
        weights = weights/torch.sum(weights)
        return weights.unsqueeze(1)
    
        # tokens = self.origin_tokenizer.convert_ids_to_tokens(token_ids)
        # weights = [len(c) -2  if c.startswith('##') else len(c) for c  in tokens]
        # count_tensor = torch.tensor(weights).pow(1/2)

        # std = torch.std(count_tensor)
        # mean_c = torch.mean(count_tensor)
        # n1 = (count_tensor-mean_c)/std
   
        # n1 = torch.softmax(n1,dim=0)

   
        # print('counts',counts)
        # print('weights',weights)
        # print('count_tensor',count_tensor)

        # count_softmax = torch.softmax(count_tensor,dim=0)
        # print(count_softmax)
        # return count_softmax.unsqueeze(0).t()
        # min_c = torch.min(count_tensor)
        # max_c =  torch.max(count_tensor)
        # n1 = (count_tensor - min_c)/(max_c - min_c)
        # return n1.unsqueeze(1)

   

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 

def rows_to_dict(rows):
    return {(r["SubjectEntity"], r["Relation"]): r for r in rows}


if __name__ == "__main__":
    tw = Token_Weight()

    origin_tokenizer = BertTokenizerFast.from_pretrained(config.bert_base_cased)

    word = 'track cyclist'
    index_list =  origin_tokenizer.encode(word)[1:-1]
    print(origin_tokenizer.convert_ids_to_tokens(index_list))
    print(tw.token_weight(index_list))
    print()

    word = 'traffic collision'
    index_list =  origin_tokenizer.encode(word)[1:-1]
    print(origin_tokenizer.convert_ids_to_tokens(index_list))
    print(tw.token_weight(index_list))
    print()


    import torch
    import torch.nn as nn

    # Define a BatchNorm layer
    batchnorm_layer = nn.InstanceNorm1d(3)  # BatchNorm for 1D data (3 values)

    # Input data (activations after convolution)
    input_data = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0]], requires_grad=True)

    # Forward pass through BatchNorm
    output_data = batchnorm_layer(input_data)

    # Print input and output
    print("Input Data:")
    print(input_data)
    print("\nOutput Data:")
    print(output_data)

