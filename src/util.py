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
from tqdm import tqdm
import pandas as pd

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
    train_data = []
    with open(data_fn, "r") as file:
        lines = file.readlines()
        for line in lines:
            train_data.append(json.loads (line))

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
    if recode_type == 'weight':
        tw = Token_Weight(origin_tokenizer)
    for entity,index in additional_token_dict.items():
        token_ids = origin_tokenizer.encode(entity,add_special_tokens=False)
      
        old_output = old_output_embedding.weight.data[token_ids,:]
        old_input = old_token_embedding.weight.data[token_ids,:]
        old_cls_bias = model.cls.predictions.bias.data[token_ids]
        old_cls_decoder_data = model.cls.predictions.decoder.weight.data[token_ids,:]

        # print("old_output",old_output.shape)
        # print("weight",weight.shape)
        if recode_type == 'weight':
            # weight = tw.token_weight(token_ids)
            # new_output = torch.multiply(old_output,weight).sum(dim=0)
            # new_output = torch.nn.functional.normalize(new_output, p=2, dim=1)
            # if len (token_ids) == 2 and print_count < 10:
            # # and Fals
            #     print_count+=1
            #     print('entity',entity)
            #     print('tokens',origin_tokenizer.convert_ids_to_tokens(token_ids))
            #     print('old_output',old_output[:,:5])
            #     print('weight',weight)
            #     print('new_output',new_output[:5])
            #     print()
            # new_input =  torch.multiply(old_input,weight).sum(dim=0)
            # new_input = torch.nn.functional.normalize(new_input, p=2, dim=1)
            # # print('old_cls_bias',old_cls_bias.shape)
            # new_cls_bias = torch.multiply(old_cls_bias,weight).sum()
            # # print('new_cls_bias',new_cls_bias.shape)
            # new_cls_decoder_data =torch.multiply(old_cls_decoder_data,weight).sum(dim=0)
            # new_cls_decoder_data = torch.nn.functional.normalize(new_cls_decoder_data, p=2, dim=1)
            def merge_embedding_1(old_embedding,weight):
                new_embedding = torch.multiply(old_embedding,weight)
                # new_embedding = torch.nn.functional.normalize(new_embedding, p=2, dim=1)
                new_embedding=new_embedding.sum(dim=0)
                new_embedding = torch.nn.functional.normalize(new_embedding,dim=-1)
                return new_embedding

            weight = tw.token_weight(entity)
            new_output = merge_embedding_1(old_output,weight)
            if len (token_ids) == 2 and print_count < 10:
            # and Fals
                print_count+=1
                print('entity',entity)
                print('tokens',origin_tokenizer.convert_ids_to_tokens(token_ids))
                print('old_output',old_output[:,:5])
                print('weight',weight)
                print('new_output',new_output[:5])
                print()
            new_input =  merge_embedding_1(old_input,weight)
            new_cls_bias = torch.multiply(old_cls_bias,weight).sum()
            # print('new_cls_bias',new_cls_bias.shape)
            new_cls_decoder_data =merge_embedding_1(old_cls_decoder_data,weight)

            
        elif recode_type == 'mean':
            new_output =   torch.mean(old_output,0,keepdim = True)
            new_output = torch.nn.functional.normalize(new_output,dim=-1)
            new_input =  torch.mean(old_input,0,keepdim = True)
            new_input = torch.nn.functional.normalize(new_input,dim=-1)
            new_cls_bias = torch.mean(old_cls_bias)
            new_cls_decoder_data = torch.mean(old_cls_decoder_data,0,keepdim = True)
            new_cls_decoder_data = torch.nn.functional.normalize(new_cls_decoder_data,dim=-1)
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
            new_output = torch.nn.functional.normalize(new_output,dim=-1)
            new_input = torch.nn.functional.normalize(new_input,dim=-1)
            new_cls_decoder_data = torch.nn.functional.normalize(new_cls_decoder_data,dim=-1)

            
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


def opt_token_layer(model:transformers.BertForMaskedLM, enhance_tokenizer, origin_tokenizer:transformers.BertweetTokenizer,recode_type):
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
 
    # old_position = model.get_position_embeddings()
    # position_dim_0, position_dim_1 = old_position.weight.shape
    # position_dim_0_new = position_dim_0+len(additional_token_dict)
    # new_position = torch.nn.Embedding(
    #      position_dim_0_new, old_embedding_dim
    # )
    # new_embeddings.padding_idx = embedding_layer.clone()
    print_count= 0
    if recode_type == 'weight':
        tw = Token_Weight(origin_tokenizer)
    for entity,index in additional_token_dict.items():
        token_ids = origin_tokenizer.encode(entity,add_special_tokens=False)
      
        old_output = old_output_embedding.weight.data[token_ids,:]
        old_input = old_token_embedding.weight.data[token_ids,:]

        # print("old_output",old_output.shape)
        # print("weight",weight.shape)
        if recode_type == 'weight':
            weight = tw.token_weight(entity)
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
     
        # new_position .weight.data[index,:] = new_position_token
    # print("old_token_embedding ", old_token_embedding.weight.data[2][:5])
    # print("new_token_embeddings ", new_input_embeddings.weight.data[2][:5])
    # print("new_output_embeddings ", new_output_embeddings.weight.data[2][:5])
    # print("old_output_embedding ", old_output_embedding.weight.data[2][:5])
    model.set_input_embeddings( new_input_embeddings)
    model.set_output_embeddings( new_output_embeddings)
    model.config.vocab_size = num_new_tokens
    model.vocab_size = num_new_tokens
    return model

class Token_Weight:
    def __init__(self,origin_tokenizer):
        self._weight:dict = json.load(open(config.token_count_fp))
        self.origin_tokenizer = origin_tokenizer

    def _token_weight(self, entity:str):
        tokens = self.origin_tokenizer.tokenize(entity)
        if entity.count(' ') > 0:
            token_ids = self.origin_tokenizer.convert_tokens_to_ids(tokens)
            return self._token_weight(token_ids)
        else:
            weight = [1/len(tokens)]*len(tokens)
            weight =  torch.tensor(weight,dtype=torch.float32)
            weight = weight.unsqueeze(1)
            return weight

    def token_weight(self, entity:str):
        token_ids = self.origin_tokenizer.encode(entity,add_special_tokens=False)
        # print('token_ids',token_ids)
        counts = [self._weight[str(tid)] if str(tid) in self._weight else 1000000 for tid in token_ids]
        count_tensor = torch.tensor(counts,dtype=torch.float32)

        # weights = count_tensor/torch.sum(count_tensor)
        # min_value, min_index = torch.min(weights,0)
        # # max_value, max_index = torch.max(weights,0)
        # weight = -1*weights
        # weight[min_index] = 1
        # return weight.unsqueeze(1)

        weights = 1/count_tensor
        weights = weights.pow(1/8)
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
        raise ValueError("Unsupported value type")
 

def rows_to_dict(rows):
    return {(r["SubjectEntity"], r["Relation"]): r for r in rows}





def adaptive_threshold(args):
    origin_result_dict = evaluate.evaluate(args.output_fn, args.valid_fn)
    predefine_fine = 'res/object_number.tsv'
    with open(predefine_fine) as f:
        topk = csv.DictReader(f, delimiter="\t")
        topk_max_dict = { row["Relation"]:eval(row['Val'])[1] for row in topk}
    threshold_initial_dict = dict()
    for k,v in topk_max_dict.items():
        threshold_initial_dict[k] = 1/float(v)
    pred_rows = util.file_read_json_line(args.output_fn)
    groud_rows = util.file_read_json_line(args.valid_fn)
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

    relation_threshold_dict = dict()
    relation_index= dict()
    # print('threshold_initial_dict',threshold_initial_dict)
    for relation, pred_list in tqdm(relation_list_pred.items()):
        groud_list = relation_list_groud[relation]
        origin_topk = topk_max_dict[relation]
        best_f1=0
        best_precision = 0
        best_recal= 0 
        origin_threshold = threshold_initial_dict[relation]
        best_threshold=0
        threshold_step = 0.01
        best_index = 100
        for i in range(1,int(0.5//threshold_step)):
            threshold = threshold_step*i
            # try_times=0
            for row in pred_list:
                score_index = 0 
                for i, score in enumerate(row['ObjectEntitiesScore']):
                    if score <= threshold:
                        score_index = i
                        break

                row['ObjectEntities'] =row['ObjectEntities'][:score_index]
                row['ObjectEntitiesID'] =row['ObjectEntitiesID'][:score_index]

            eval_dict = evaluate.evaluate_list(groud_list, pred_list)[relation]
            f1 = eval_dict['f1']
            p = eval_dict['p']
            r = eval_dict['r']
            if f1> best_f1:
                best_f1=f1
                best_threshold =threshold
                best_precision= p 
                best_recal= r
                best_index= score_index
                # try_times = 0
            # else:
            #     try_times+=1
            #     if try_times > 3:
            #         break

  
        relation_index[relation] = best_index
        relation_threshold_dict[relation] = best_threshold
    
        origin_result_dict[relation]["best_precision"]=best_precision
        origin_result_dict[relation]["best_recal"]=best_recal
        origin_result_dict[relation]["best_f1"]=best_f1
        origin_result_dict[relation]["best_threshold"]=best_threshold

    pred_rows = util.file_read_json_line(args.output_fn)
    for row in pred_rows:
        relation = row[config.KEY_REL]
        row[config.KEY_OBJS] = row[config.KEY_OBJS][:relation_index[relation]]
        row[config.KEY_OBJS_ID] = row[config.KEY_OBJS_ID][:relation_index[relation]]
    util.file_write_json_line(args.output_fn+'.ths',pred_rows)
        #origin_result_dict[relation]["origin_threshold"]=origin_threshold

    with open(rel_thres_fn,'w') as f:
        json.dump(relation_threshold_dict,f,indent = 2)
    origin_result_dict["Average"]["best_f1"] =  sum([x["best_f1"] if "best_f1" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    origin_result_dict["Average"]["best_precision"] =  sum([x["best_precision"] if "best_precision" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    origin_result_dict["Average"]["best_recal"] =  sum([x["best_recal"] if "best_recal" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    result_dict = {
        "args":args.__dict__,
        "metric":origin_result_dict
        }
    util.file_write_json_line(config.RESULT_FN, [result_dict],'auto')
    scores_per_relation_pd = pd.DataFrame(origin_result_dict)
    print(scores_per_relation_pd.transpose().round(3).to_string(max_cols=12))



    # for relation, v in best_topk_dict.items():
    #     print(relation,v[1],v[0], topk_max_dict[relation] )
    # # print(json.dumps(best_topk_dict, indent=4))
    # average_f1 = sum(map(lambda x:x[1], best_topk_dict.values()))/ len(best_topk_dict)
    # print("average_f1",average_f1)
    #  


if __name__ == "__main__":
    origin_tokenizer = BertTokenizerFast.from_pretrained(config.bert_base_cased)

    tw = Token_Weight(origin_tokenizer)


    word = 'United States of America'
    # index_list =  origin_tokenizer.encode(word)[1:-1]
    # print(origin_tokenizer.convert_ids_to_tokens(index_list))
    print(tw.token_weight(word))
    print()

    # word = 'traffic collision'
    # index_list =  origin_tokenizer.encode(word)[1:-1]
    # print(origin_tokenizer.convert_ids_to_tokens(index_list))
    # print(tw.token_weight(index_list))
    # print()
