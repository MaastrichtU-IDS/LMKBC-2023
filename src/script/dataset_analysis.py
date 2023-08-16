import json
import os
import random
import sys
import pandas as pd

from sympy import to_cnf



parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)
import evaluate as evaluate
from glob import glob

import config
import util
import transformers
from tqdm import tqdm


with open(f'{config.RES_DIR}/tokenizer/bert/added_tokens.json') as f:
    entity_dic_jt = json.load(f)

train_line = util.file_read_json_line(config.TRAIN_FN)
valid_line = util.file_read_json_line(config.VAL_FN)
all_gold_lines = train_line + valid_line

origin_tokenizer = transformers.AutoTokenizer.from_pretrained(config.bert_base_cased)
# enhance_tokenizer = transformers.AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)

def refresh_tokenizer(entity_set:set):
    origin_tokenizer.add_tokens(list(entity_set))
    origin_tokenizer.save_pretrained(config.TOKENIZER_PATH)



def count_object():
    null_count = 0
    for line in all_gold_lines:
        obj_list = line[config.KEY_OBJS]
        if len(obj_list) == 0:
            null_count += 1
    print("null_count", null_count)


def count_emapt():
    empty_count = 0
    for line in all_gold_lines:
        obj_list = line[config.KEY_OBJS]
        for obj in obj_list:
            if obj == config.EMPTY_TOKEN:
                empty_count += 1
    print("empty_count", empty_count)


def count_emapt_and_other():
    empty_count = 0
    for line in all_gold_lines:
        obj_list = line[config.KEY_OBJS]
        if '' in obj_list and len(obj_list) > 1:
            empty_count += 1
    print("empty_count", empty_count)


def get_all_entity(fn = None):
    if fn is not None:
        lines = util.file_read_json_line(fn)
    else:
        lines = all_gold_lines
    entity_set = set()
    for row in lines:
        object_entities = row['ObjectEntities']
        subject = row["SubjectEntity"]
        entity_set.add(subject)
        entity_set.update(object_entities)
    return entity_set


def collect_entity():
    entity_fn = f'{config.RES_DIR}/additional_corpus/entity.txt'
    # entity_1_set = [t.replace("\"", "") for t in list(entity_set)]
    util.file_write_line(entity_fn, list(entity_dic_jt.keys()), mode='w')

def negave_vocabulary():
    true_entity = get_all_entity()
    
    output_fm_train = "output/filled-mask/filled-mask-train.jsonl"
    output_fm_valid = "output/filled-mask/filled-mask-valid.jsonl"
    predict_entity_train = get_all_entity(output_fm_train)
    predict_entity_valid = get_all_entity(output_fm_valid)
    false_entity =(predict_entity_train| predict_entity_valid) -  true_entity
    print(false_entity)
    entity_fn = f'data/negave_vocabulary.json'
    with open(entity_fn,'w') as f:
        json.dump(list(false_entity),f)

def print_relation_dict():
    relation_dict=dict()
    for line in all_gold_lines:
        relation = line[config.KEY_REL]
        if relation not in relation_dict:
            relation_dict[relation] =["person","person"]
        if len(relation_dict) ==21:
            print(json.dumps(relation_dict,indent =4))
            break

def is_wiki_id(aStr:str):
    return  (aStr.startswith('Q') and str.isdigit(aStr[1:]))


def collect_entity_pretrain():
    # exclude_entities = {}
    # exclude_entities = {"Person"}
    sub_type ={s for s,_ in  config.relation_entity_type_dict.values()}
    obj_type ={o for s, o in  config.relation_entity_type_dict.values()}
    # s_type = sub_type - obj_type
    # exclude_entities = exclude_entities | s_type
    # exclude_entities = {'Series', 'Compound', 'Band', 'Person'}
    exclude_entities = {'Compound', 'Band', 'Person'}
    print("exclude_entities",exclude_entities)
    exclude_subject={"RiverBasinsCountry"}
    give_up_relation = {"SeriesHasNumberOfEpisodes" }
    silver_lines = []
    file_name_list = glob(f'{config.RES_DIR}/silver/*.jsonl', recursive=True)
    print("file_name_list", len(file_name_list))
    for filename in file_name_list:
        with open(filename) as f:
            lines = util.file_read_json_line(filename)
            silver_lines.extend(lines)
    gold_dict, gold_entity_sentence = count_entity_distribution(all_gold_lines,
                                          exclude_entities=exclude_entities,
                                          merge_subject=True
                                          )
    siler_dict,silver_entity_sentence = count_entity_distribution(silver_lines,
                                           exclude_entities = exclude_entities,
                                            merge_subject=True,
                                            exclude_subject=exclude_subject,
                                            give_up_relation=give_up_relation
                                            )

    for k, v in siler_dict.items():
        siler_dict[k] =set(filter(lambda x: not is_wiki_id(x), v) )
    if 'Number' in siler_dict:
        siler_dict['Number'] = set(filter(lambda x: str.isdigit(x), siler_dict['Number']))
    # print(siler_dict['River'])
    result_dict=dict()
    for k in gold_dict.keys():
        if k in siler_dict:
            result_dict[k] = gold_dict[k] | siler_dict[k]
        else:
            result_dict[k] = gold_dict[k]

    entity_type_dict = dict()
    for k, v in result_dict.items():
        for vi in v:
            entity_type_dict[vi] = k
    
    entity_set = set.union(* [set(e) for e in result_dict.values()])
    entity_remove = set()
    for e in entity_set:
            for m in entity_set:
                if e == m:
                    continue
                if e in m.split() and entity_type_dict[e] == entity_type_dict[m]:
                    # print(e," , ",m, " , " ,entity_type_dict[e])
                    entity_remove.add(m)

    entity_set -= entity_remove
    for k, v in result_dict.items():
        v-=entity_remove
    entity_sentence = gold_entity_sentence|silver_entity_sentence
    less_type=set()
    for e in entity_remove:
        if len(e) > 10:
            print(e, entity_type_dict[e])
    # for e in entity_set:
    #     if len(e) < 5:
    #         less_type.add(entity_type_dict[e])
    #         print(e, entity_type_dict[e])
    #         print(entity_sentence[e])
    #         print()
    print(less_type)
    print(result_dict.keys() - less_type)
    # display_entity_dict(gold_dict,remove_tokenizer=True)
    # display_entity_dict(result_dict,remove_tokenizer=True)
    display_entity_dict(result_dict)
    # count_entity_distribution(all_lines)



    entity_fp = f'{config.RES_DIR}/entity_for_pretrain.json'
    for k, v in result_dict.items():
        result_dict[k] = list(v)
    with open(entity_fp, mode='w') as f:
        json.dump(result_dict,f,indent=2,sort_keys=True)


    # refresh_tokenizer(entity_set)



def merge_dict(dict_list):
    result_dict=dict()
    for aDict in dict_list:
        for k in aDict.keys():
            if k in result_dict:
                result_dict[k] = set(aDict[k]) |set(result_dict[k])
            else:
                result_dict[k] = aDict[k]
    return result_dict
        

def display_entity_contains(gold_dict, siler_dict,entity_type_dict):

    entity_type_gold_dict = dict()
    for k, v in gold_dict.items():
        for vi in v:
            entity_type_gold_dict[vi] = k
    entity_type_silver_dict = dict()
    for k, v in siler_dict.items():
        for vi in v:
            entity_type_silver_dict[vi] = k
    source_dict= dict()
    
    for k, v in entity_type_dict.items():
        source_dict[k] = ''
        if k in entity_type_gold_dict:
            source_dict[k] += " gold "
        if k in entity_type_silver_dict:
            source_dict[k] += " silver "

    entity_remove = set()
    short_count_dict=dict()
    long_count_dict = dict()
    for e in entity_type_dict.keys():
            for m in entity_type_dict.keys():
                if e == m:
                    continue
                if e in m.split() and entity_type_dict[e] == entity_type_dict[m]:
                    # print(e, " , ",source_dict[e], " , ", m, " , " ,entity_type_dict[e], " , ", source_dict[m])
                    if source_dict[e] not in short_count_dict:
                        short_count_dict[source_dict[e]] =1
                    else:
                        short_count_dict[source_dict[e]]+=1

                    if source_dict[m] not in long_count_dict:
                        long_count_dict[source_dict[m]] =1
                    else:
                        long_count_dict[source_dict[m]] +=1
    print("short_count_dict", short_count_dict)
    print("long_count_dict", long_count_dict)


def get_silver_lines():
    silver_lines=[]
    file_name_list = glob(f'{config.RES_DIR}/silver/*.jsonl', recursive=True)
    print("file_name_list", len(file_name_list))
    for filename in file_name_list:
        with open(filename) as f:
            lines = util.file_read_json_line(filename)
            silver_lines.extend(lines)
    return silver_lines

def collect_entity_for_tokenizer():
    exclude_entities = {"Person"}
    sub_type ={s for s,_ in  config.relation_entity_type_dict.values()}
    obj_type ={o for s, o in  config.relation_entity_type_dict.values()}
    sub_only_type = sub_type - obj_type
    # silver_lines = get_silver_lines()
    test_silver_fp = 'res/test_silver.jsonl'
    silver_lines = util.file_read_json_line(test_silver_fp)
    print('silver_lines', len(silver_lines))
    gold_dict, gold_entity_sentence = count_entity_distribution(all_gold_lines,
                                        #   exclude_entities=exclude_entities,
                                        #   merge_subject=True,
                                        remove_tokenizer=True,
                                        include_entities=obj_type,
                                          )
    siler_dict,silver_entity_sentence = count_entity_distribution(silver_lines,
                                        #    exclude_entities = exclude_entities,
                                            # merge_subject=False,
                                            # exclude_subject=exclude_subject,
                                            # give_up_relation=give_up_relation
                                              remove_tokenizer=True,
                                               include_entities=obj_type,
                                            )
    print('siler_dict', siler_dict.keys())
    result_dict=merge_dict([gold_dict,siler_dict])
    print('result_dict', result_dict.keys())
    # for sub in sub_only_type:
    #     del result_dict[sub]
    # print(sub_only_type)
    display_entity_dict(result_dict)
    # entity_fp = f'{config.RES_DIR}/entity_for_tokenizer.json'
    # for k, v in result_dict.items():
    #     result_dict[k] = list(v)
    # with open(entity_fp, mode='w') as f:
    #     json.dump(result_dict,f,indent=2,sort_keys=True)
    entity_set = set.union(* [set(e) for e in result_dict.values()])
    refresh_tokenizer(entity_set)
    # multi_thread_entity(entity_set)
    # collection_ids(entity_set)
    # entity_id_dict = get_entity_ids(entity_set)


def collect_entity_for_pretrain():
    test_silver_fp = 'res/test_silver.jsonl'
    exclude_entities = {}
    exclude_entities = {"Person"}
    test_row = util.file_read_json_line(config.test_fp)
    sub_type ={s for s,_ in  config.relation_entity_type_dict.values()}
    obj_type ={o for s, o in  config.relation_entity_type_dict.values()}
    all_type = sub_type|obj_type
    sub_only_type = sub_type - obj_type
    # exclude_entities = exclude_entities | s_type
    exclude_entities = {'Series', 'Person','Number','Instrument','Position','Profession','Band','Company'}
    include_entities ={'State','Country','Language'}
    # exclude_entities = {'Person'}
    print("exclude_entities",exclude_entities)
    exclude_subject={"RiverBasinsCountry"}
    give_up_relation = {"SeriesHasNumberOfEpisodes" }
    silver_lines = []
    file_name_list = glob(f'{config.RES_DIR}/silver/*.jsonl', recursive=True)
    print("file_name_list", len(file_name_list))
    for filename in file_name_list:
        with open(filename) as f:
            lines = util.file_read_json_line(filename)
            silver_lines.extend(lines)
    test_silver_line = util.file_read_json_line(test_silver_fp)
    gold_dict, gold_entity_sentence = count_entity_distribution(valid_line,
                                          merge_subject=True,
                                          include_entities=all_type,
                                          )
    silver_dict,silver_entity_sentence = count_entity_distribution(test_row,
                                            merge_subject=True,
                                            include_entities=all_type,
                                            # give_up_relation=give_up_relation
                                            )
    test_silver_dict,silver_entity_sentence = count_entity_distribution(test_silver_line,
                                            merge_subject=True,
                                            include_entities=all_type,
                                            # give_up_relation=give_up_relation
                                            )
    
    result_dict=merge_dict([gold_dict,silver_dict,test_silver_dict])

    # entity_type_dict = dict()
    # for k, v in result_dict.items():
    #     for vi in v:
    #         entity_type_dict[vi] = k

    # display_entity_contains(gold_dict,siler_dict,entity_type_dict)
    # entity_set = set.union(* [set(e) for e in result_dict.values()])

    # entity_sentence = gold_entity_sentence|silver_entity_sentence
    # less_type=set()
    # for e in entity_remove:
    #     if len(e) > 10:
    #         print(e, entity_type_dict[e])
    # for e in entity_set:
    #     if len(e) < 5:
    #         less_type.add(entity_type_dict[e])
    #         print(e, entity_type_dict[e])
    #         print(entity_sentence[e])
    #         print()
    # print(less_type)
    # print(result_dict.keys() - less_type)
    # display_entity_dict(gold_dict,remove_tokenizer=True)
    # display_entity_dict(result_dict,remove_tokenizer=True)
    display_entity_dict(result_dict)
    # count_entity_distribution(all_lines)
    entity_fp = f'{config.RES_DIR}/entity_for_pretrain.json'
    for k, v in result_dict.items():
        result_dict[k] = list(v)
    with open(entity_fp, mode='w') as f:
        json.dump(result_dict,f,indent=2,sort_keys=True)

def print_relation_entity_type():
    items = sorted(config.relation_entity_type_dict.items(),key = lambda x: x[0])
    for k,v in items:
        s,o = v
        print(f'{k} & {s} & {o}')
    
 
def collect_entity_for_pretrain_test():
    exclude_entities = {}
    exclude_entities = {"Person"}
    sub_type ={s for s,_ in  config.relation_entity_type_dict.values()}
    obj_type ={o for s, o in  config.relation_entity_type_dict.values()}
    all_type = sub_type| obj_type
    sub_only_type = sub_type - obj_type
    # exclude_entities = exclude_entities | s_type
    exclude_entities = {'Series', 'Person','Number','Instrument','Position','Profession','Band','Company'}
    include_entities ={'State','Country','Language'}
    # exclude_entities = {'Person'}
    print("exclude_entities",exclude_entities)
    exclude_subject={"RiverBasinsCountry"}
    give_up_relation = {"SeriesHasNumberOfEpisodes" }
    silver_lines = []
    file_name_list = glob(f'{config.RES_DIR}/silver/*.jsonl', recursive=True)
    print("file_name_list", len(file_name_list))
    for filename in file_name_list:
        with open(filename) as f:
            lines = util.file_read_json_line(filename)
            silver_lines.extend(lines)
    test_lines = util.file_read_json_line(config.test_silver_fp)
    result_dict, gold_entity_sentence = count_entity_distribution(test_lines,
                                          merge_subject=True,
                                          include_entities=all_type,
                                          merge_object=True
                                          )

    display_entity_dict(result_dict)
    # count_entity_distribution(all_lines)
    entity_fp = f'{config.RES_DIR}/entity_for_pretrain_test.json'
    for k, v in result_dict.items():
        result_dict[k] = list(v)
    with open(entity_fp, mode='w') as f:
        json.dump(result_dict,f,indent=2,sort_keys=True)

    
    
 


def display_entity_dict(entity_dict):
    all_size=0  
    for k,v in entity_dict.items():
 
        print(f"{k} & {len(v)} \\\\")
        all_size+=len(v)
 
    print("entity size is ",all_size)
    print()

def count_entity_distribution(all_lines,
                              remove_tokenizer=True,
                              merge_subject = False,
                               merge_object = True,
                              include_entities={},
                              give_up_relation={},

                              ):
    entity_dict =dict()
    entity_sentence = dict()
    for line in all_lines:
        relation = line[config.KEY_REL]
        if relation in give_up_relation:
            # print(relation)
            continue
    
        objs = line[config.KEY_OBJS]
        sub = line[config.KEY_SUB]
        sub_type, obj_type = config.relation_entity_type_dict[relation]
        if merge_subject:
            # print(relation, exclude_subject)
            if  sub_type  in include_entities:
                if sub_type not in  entity_dict:
                    entity_dict[sub_type] = set()
                entity_dict[sub_type].add(sub)
                entity_sentence[sub] = line
        if merge_object:
            if obj_type  in include_entities:
                if obj_type not in  entity_dict:
                    entity_dict[obj_type] = set()         
                entity_dict[obj_type].update(objs)
                for obj in objs:
                    entity_sentence[obj] = line
        # if 'Number' in entity_dict:
        #     entity_dict['Number'] = set(filter(lambda x: str.isdigit(x), entity_dict['Number']))
        # for k, v in entity_dict.items():
        #     entity_dict[k] =set(filter(lambda x: not is_wiki_id(x), v) )

    if remove_tokenizer:
        for k,v in entity_dict.items():
            entity_dict[k]-= origin_tokenizer.vocab.keys()
            if '' in entity_dict[k]:
                entity_dict[k].remove('')
            
            # entity_set = set()
            # for entity in entity_dict[k]:
            #     if r'\u' in entity:
            #         entity = entity.replace(r'\u',r'\\u')
            #         entity  = entity.encode('utf8').decode('unicode-escape')  
            #     entity_set.add(entity)
            # entity_dict[k] = entity_set
        entity_remove = set()

    # for e in entity_set:
    #         for m in entity_set:
    #             if e == m:
    #                 continue
    #             if e in m.split() and entity_type_dict[e] == entity_type_dict[m]:
    #                 # print(e," , ",m, " , " ,entity_type_dict[e])
    #                 entity_remove.add(m)

    # entity_set -= entity_remove
    # for k, v in result_dict.items():
    #     v-=entity_remove

    return entity_dict,entity_sentence

def collection_ids(entity_set):
    entity_count=0
    for entity in tqdm(entity_set):
        id = util.disambiguation_baseline(entity)
        if entity_count% 100 ==0:
            util.save_entity_id()
    util.save_entity_id()

import requests

def get_entity_ids(entity_labels):
    entity_labels= list(entity_labels)[:10]
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "props": "info",
        "languages": "en",
        "titles": "|".join(entity_labels),
    }
    entity_id_dict = dict()
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        print("data",data)
        entities = data.get("entities", {})
        for label in entity_labels:
            entity_data = entities.get(label, {})
            if "title" in entity_data:
                entity_id = entity_data["title"]
                entity_id_dict[label] = entity_id
        
    else:
        print(f"Error: Request failed with status code {response.status_code}")
    return entity_id_dict

def multi_thread_entity(entity_set):
    import requests
    from concurrent.futures import ThreadPoolExecutor,as_completed
    # entity_set=list(entity_set)[10000:]
    util.local_cache.clear()
    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = [(entity,executor.submit(util.disambiguation_baseline, entity)) for entity in tqdm(entity_set)]
        futures_submit = list(map(lambda x:x[1],futures))
        future_entity_dict = dict()
        for entity, future in futures:
            future_entity_dict[future] = entity
        entity_dict = dict()
        for future in tqdm(as_completed(futures_submit),total = len(entity_set)):
            # print(future_entity_dict[future], future.result())
        # for future in futures:
            # result = future.result()
            entity_dict[future_entity_dict[future]]=future.result()
    util.save_entity_id()
    # with open(f'{config.RES_DIR}/entity_id.json','w') as f:
    #     json.dump(entity_dict,f,indent = 2)
    
def same_id_numer():
    with open(f'{config.RES_DIR}/entity_id.json') as f:
        entity_id = json.load(f)
    id_count= dict()
    for e, i in entity_id.items():
        if i not in id_count:
            id_count[i] = 1
        else:
            id_count[i] +=1
    redunt_number = 0
    for i, c in id_count.items():
        if c>1:
            redunt_number+=c
    print("redunt_number",redunt_number)


def tokenize():
    entity_list =["Japan", "People's Republic of China", "North Korea"]
    for e in entity_list:
        print(e,  origin_tokenizer.tokenize(e))
    print(origin_tokenizer.mask_token)


def rows_to_dict(rows):
    return {(r["SubjectEntity"], r["Relation"]): r for r in rows}

def according_test():
    test_row = util.file_read_json_line(config.test_fp)
    val_row = util.file_read_json_line(config.VAL_FN)
    silver_lines = []
    file_name_list = glob(f'{config.RES_DIR}/silver/*.jsonl', recursive=True)
    print("file_name_list", len(file_name_list))
    for filename in file_name_list:
        with open(filename) as f:
            lines = util.file_read_json_line(filename)
            silver_lines.extend(lines)
    test_dict = rows_to_dict(test_row)
    val_dict = rows_to_dict(val_row)
    silver_dict = rows_to_dict(silver_lines)
    result_lines= []
    for test_key in test_dict.keys():
        if test_key in silver_dict:
            result_lines.append(silver_dict[test_key])
        
    # util.file_write_json_line('res/test_silver.jsonl',result_lines)
    test_silver_fp = 'res/test_silver.jsonl'
    util.file_write_json_line('res/test_silver.jsonl',result_lines)
    print(len(result_lines))


def token_weight():
    corpus_fp = 'res/wikidata/Country-Language-State/filter.json'
    lines = util.file_read_json_line(corpus_fp)
    token_count = dict()
    for line in tqdm(lines):
        sentence = line['sentence']
        tokens_ids = origin_tokenizer.encode(sentence)
        for t in tokens_ids:
            if t not in token_count:
                token_count[t] = 0
            token_count[t]+=1
    token_count_fp = config.token_count_fp
    json.dump( token_count,open(token_count_fp,'w'),indent=2)

def case_study():
    aim_fp = 'output/filled-mask/token_recode_std/filled-mask-valid.jsonl'
    std_lines = util.file_read_json_line(aim_fp)
    baseline_line = 'output/filled-mask/fine-tune/filled-mask-valid.jsonl'
    bl_lines = util.file_read_json_line(baseline_line)
    std_dict = util.rows_to_dict(std_lines)
    for bl in bl_lines:
        std_line = std_dict[bl[config.KEY_SUB],bl[config.KEY_REL]]
        if sum( bl['ObjectLabels']) < sum(std_line['ObjectLabels']):
            print()
            print(bl)
            print(std_line)


def print_result():
    aim_fp = 'output/filled-mask/pretrain-val_test/filled-mask-valid.jsonl'
    std_lines = util.file_read_json_line(aim_fp)
    baseline_line = 'testrun-bert.jsonl'
    bl_lines = util.file_read_json_line(baseline_line)
    std_result = evaluate.evaluate_list(valid_line,std_lines)
    bl_result = evaluate.evaluate_list(valid_line,bl_lines)
    # std_dict = util.rows_to_dict(std_lines)
    for bl in bl_result.keys():
        std_line = std_result[bl]
        bl_result[bl]['p1'] = std_line['p']
        bl_result[bl]['r1'] = std_line['r']
        bl_result[bl]['f11'] = std_line['f1']
    pd_result = pd.DataFrame(bl_result)
    print(pd_result.transpose().round(4).to_string(max_cols=12,decimal='&'))

if __name__ == "__main__":
    # collect_entity_for_pretrain()
    # collect_entity_for_pretrain_test()
    collect_entity_for_tokenizer()
    # collection_ids()
    # same_id_numer()
    # tokenize()
    # according_test()
    # token_weight() 
    # print(sum([0,0,1]))
    # print_result()
    # print_relation_entity_type()

